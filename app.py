import base64

import streamlit as st
import tiktoken
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


load_dotenv()
load_dotenv("env", override=False)


CLARIFY_MODEL = "gpt-5.4-mini"
FINAL_MODEL = "gpt-5.5"

MODEL_PRICES = {
    CLARIFY_MODEL: {
        "input": 0.75 / 1_000_000,
        "output": 4.5 / 1_000_000,
    },
    FINAL_MODEL: {
        "input": 5 / 1_000_000,
        "output": 30 / 1_000_000,
    },
}

CLARIFY_PROMPT = """
너는 의학적 권위를 지니고 있는 AI야. 내가 말하는 증상에 대해서 자세하게 분석해줘.

다만 너는 의사를 대체하지 않는다.
확정 진단처럼 말하지 말고, 사용자의 증상을 더 잘 파악하기 위한 문진을 해줘.

사진 사전 분석 참고가 제공되면 그 내용은 이미 확인된 정보로 취급해.
예를 들어 사진 사전 분석에 "발바닥", "피", "출혈", "상처"가 나오면
"어디에 피가 나나요?"처럼 같은 내용을 다시 묻지 마.
대신 언제 다쳤는지, 얼마나 피가 나는지, 통증 정도, 이물질 여부, 파상풍 접종 여부처럼
사진만으로 알 수 없는 정보를 물어봐.

반드시 확인해야 하는 정보:
- 어디가 아픈지
- 어떻게 아픈지
- 통증이나 불편감 강도
- 언제부터 아팠는지
- 좋아지거나 나빠지는 상황
- 함께 나타나는 증상
- 지병
- 먹고 있는 약이나 알레르기
- 나이, 성별, 임신 가능성 같은 기본 정보

아직 정보가 부족하면 가장 중요한 질문 1~3개만 해줘.
정보가 충분하거나 응급 신호가 있으면 최종 분석을 진행한다고 말해줘.

응급 신호 예:
흉통, 호흡곤란, 한쪽 마비, 말 어눌함, 의식 저하, 극심한 두통, 심한 출혈, 심한 알레르기 반응

답변 맨 마지막 줄에는 반드시 아래 둘 중 하나를 붙여줘.
[FINAL_READY: YES]
[FINAL_READY: NO]
"""

FINAL_PROMPT = """
너는 의학적 권위를 지니고 있는 AI야. 내가 말하는 증상에 대해서 자세하게 분석해줘.

아래 대화 내용과 선택적으로 첨부된 사진을 보고 최종 안내를 작성해줘.
확정 진단처럼 말하지 말고 가능성 중심으로 말해줘.
응급 신호가 있으면 119 또는 응급실 방문을 우선 안내해줘.
처방약 용량, 약 시작/중단, 전문 처치는 지시하지 마.

아래 형식으로 한국어로 답해줘.

## 요약

## 가능한 질환 후보

## 응급도

## 추천 진료과

## 병원 방문 전 가능한 보존적 조치

## 바로 진료가 필요한 경고 신호

## 의사에게 전달하면 좋은 정보
"""

IMAGE_ANALYSIS_PROMPT = """
첨부된 증상 사진을 먼저 확인해줘.
확정 진단은 하지 말고, 사진에서 보이는 객관적인 특징만 한국어로 정리해줘.
이 내용은 다음 단계에서 gpt-5.4-mini가 문진 답변을 만들 때 참고할 자료야.
gpt-5.4-mini가 같은 질문을 반복하지 않도록, 확인 가능한 부위와 눈에 보이는 증상을 명확히 적어줘.

정리할 내용:
- 확인 가능한 신체 부위
- 피, 출혈, 상처, 발진, 붓기, 색 변화처럼 눈에 보이는 특징
- 사진만으로는 알 수 없는 점
- 추가로 물어보면 좋은 질문
"""


def init_page():
    st.set_page_config(page_title="무슨 병인지 알려줘", page_icon="🩺")
    st.title("무슨 병인지 알려줘")
    st.caption("증상을 입력하면 AI가 추가 문진 후 가능한 질환과 진료 방향을 안내합니다.")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    if clear_button:
        st.session_state.message_history = []
        st.session_state.image_data_url = None
        st.session_state.image_bytes = None
        st.session_state.image_name = None
        st.session_state.image_analysis = None
        st.session_state.final_result = None
        st.session_state.is_finalized = False
        st.session_state.costs = {
            CLARIFY_MODEL: {"input": 0, "output": 0},
            FINAL_MODEL: {"input": 0, "output": 0},
        }

    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    if "image_data_url" not in st.session_state:
        st.session_state.image_data_url = None
    if "image_bytes" not in st.session_state:
        st.session_state.image_bytes = None
    if "image_name" not in st.session_state:
        st.session_state.image_name = None
    if "image_analysis" not in st.session_state:
        st.session_state.image_analysis = None
    if "final_result" not in st.session_state:
        st.session_state.final_result = None
    if "is_finalized" not in st.session_state:
        st.session_state.is_finalized = False
    if "costs" not in st.session_state:
        st.session_state.costs = {}

    for model in [CLARIFY_MODEL, FINAL_MODEL]:
        if model not in st.session_state.costs:
            st.session_state.costs[model] = {"input": 0, "output": 0}


def encode_image(uploaded_file):
    image_bytes = uploaded_file.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_type = uploaded_file.type or "image/jpeg"

    st.session_state.image_data_url = f"data:{image_type};base64,{image_base64}"
    st.session_state.image_bytes = image_bytes
    st.session_state.image_name = uploaded_file.name


def show_uploaded_image():
    if st.session_state.image_bytes:
        with st.chat_message("user"):
            st.markdown("첨부한 증상 사진")
            st.image(
                st.session_state.image_bytes,
                caption=st.session_state.image_name,
                use_container_width=True,
            )


def init_chain():
    llm = ChatOpenAI(model=CLARIFY_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CLARIFY_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            (
                "user",
                "사진 사전 분석 참고:\n{image_analysis}\n\n"
                "위 사진 사전 분석에 나온 부위와 눈에 보이는 증상은 다시 묻지 말고 이미 확인된 정보로 사용해.\n\n"
                "사용자 증상:\n{user_input}",
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


def parse_final_ready(response):
    if "[FINAL_READY: YES]" in response:
        clean_response = response.replace("[FINAL_READY: YES]", "").strip()
        return clean_response, True

    clean_response = response.replace("[FINAL_READY: NO]", "").strip()
    return clean_response, False


def make_conversation_text():
    lines = []

    if st.session_state.image_analysis:
        lines.append(f"사진 사전 분석: {st.session_state.image_analysis}")

    for msg in st.session_state.message_history:
        if msg["role"] == "user":
            lines.append(f"사용자: {msg['content']}")
        else:
            lines.append(f"AI: {msg['content']}")

    return "\n".join(lines)


def analyze_image_first(user_input):
    llm = ChatOpenAI(model=FINAL_MODEL, temperature=0)

    user_content = [
        {
            "type": "text",
            "text": f"사용자의 최초 증상 입력입니다.\n\n{user_input}",
        },
        {
            "type": "image_url",
            "image_url": {"url": st.session_state.image_data_url},
        },
    ]

    response = llm.invoke(
        [
            ("system", IMAGE_ANALYSIS_PROMPT),
            ("user", user_content),
        ]
    )

    add_cost(FINAL_MODEL, IMAGE_ANALYSIS_PROMPT + user_input, response.content)
    return response.content


def run_final_analysis():
    llm = ChatOpenAI(model=FINAL_MODEL, temperature=0)
    conversation_text = make_conversation_text()

    user_content = [
        {
            "type": "text",
            "text": f"지금까지의 문진 대화입니다.\n\n{conversation_text}",
        }
    ]

    if st.session_state.image_data_url:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": st.session_state.image_data_url},
            }
        )

    response = llm.invoke(
        [
            ("system", FINAL_PROMPT),
            ("user", user_content),
        ]
    )

    add_cost(FINAL_MODEL, FINAL_PROMPT + conversation_text, response.content)
    return response.content


def get_message_counts(text, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def add_cost(model, input_text, output_text):
    input_tokens = get_message_counts(input_text, model)
    output_tokens = get_message_counts(output_text, model)

    st.session_state.costs[model]["input"] += input_tokens
    st.session_state.costs[model]["output"] += output_tokens


def calc_and_display_costs():
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 토큰 비용")

    total_cost = 0

    for model in [CLARIFY_MODEL, FINAL_MODEL]:
        input_tokens = st.session_state.costs[model]["input"]
        output_tokens = st.session_state.costs[model]["output"]

        input_cost = input_tokens * MODEL_PRICES[model]["input"]
        output_cost = output_tokens * MODEL_PRICES[model]["output"]
        model_cost = input_cost + output_cost
        total_cost += model_cost

        st.sidebar.markdown(f"**{model}**")
        st.sidebar.markdown(f"- Input tokens: {input_tokens:,}")
        st.sidebar.markdown(f"- Output tokens: {output_tokens:,}")
        st.sidebar.markdown(f"- Cost: ${model_cost:.5f}")

    st.sidebar.markdown(f"**Total Cost: ${total_cost:.5f}**")
    st.sidebar.caption("텍스트 기준 추정 비용입니다. 이미지 토큰 비용은 제외했습니다.")

    if st.session_state.image_analysis:
        with st.sidebar.expander("사진 사전 분석 확인"):
            st.write(st.session_state.image_analysis)


def main():
    init_page()
    init_messages()
    calc_and_display_costs()

    image_box = st.empty()

    if not st.session_state.message_history:
        with image_box.container():
            st.markdown("#### 증상 사진 (선택)")
            uploaded_file = st.file_uploader(
                "첫 증상 입력 전에 사진을 올릴 수 있습니다.",
                type=["png", "jpg", "jpeg", "webp"],
            )

            if uploaded_file:
                encode_image(uploaded_file)
                st.image(
                    st.session_state.image_bytes,
                    caption=f"업로드됨: {st.session_state.image_name}",
                    use_container_width=True,
                )

    for index, msg in enumerate(st.session_state.message_history):
        if index == 0:
            show_uploaded_image()

        st.chat_message(msg["role"]).markdown(msg["content"])

    if st.session_state.final_result:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.final_result)

    if st.session_state.is_finalized:
        st.success("최종 분석이 완료되었습니다. 새 상담은 Clear Conversation을 눌러 시작해 주세요.")
        return

    if st.session_state.message_history:
        if st.button("최종 분석하기", use_container_width=True):
            with st.spinner("최종 분석을 만드는 중..."):
                final_result = run_final_analysis()

            st.session_state.final_result = final_result
            st.session_state.is_finalized = True
            st.rerun()

    user_input = st.chat_input("증상을 자세히 적어주세요.")

    if user_input:
        image_box.empty()

        show_uploaded_image()
        st.chat_message("user").markdown(user_input)
        is_first_message = len(st.session_state.message_history) == 0
        st.session_state.message_history.append({"role": "user", "content": user_input})

        if is_first_message and st.session_state.image_data_url:
            with st.spinner("사진을 먼저 분석하는 중..."):
                st.session_state.image_analysis = analyze_image_first(user_input)

        chain = init_chain()

        history = []
        for msg in st.session_state.message_history[:-1]:
            history.append((msg["role"], msg["content"]))

        with st.spinner("증상을 확인하는 중..."):
            response = chain.invoke(
                {
                    "history": history,
                    "user_input": user_input,
                    "image_analysis": st.session_state.image_analysis or "첨부된 사진이 없습니다.",
                }
            )

        clean_response, final_ready = parse_final_ready(response)
        add_cost(
            CLARIFY_MODEL,
            CLARIFY_PROMPT + make_conversation_text() + (st.session_state.image_analysis or ""),
            response,
        )

        st.chat_message("assistant").markdown(clean_response)
        st.session_state.message_history.append(
            {
                "role": "assistant",
                "content": clean_response,
            }
        )

        if final_ready:
            with st.spinner("최종 분석을 만드는 중..."):
                final_result = run_final_analysis()

            st.session_state.final_result = final_result
            st.session_state.is_finalized = True

        st.rerun()


main()

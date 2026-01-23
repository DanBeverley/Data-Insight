"""Model warmup utilities."""

from langchain_core.messages import HumanMessage

from data_scientist_chatbot.app.core.agent_factory import (
    create_brain_agent,
    create_hands_agent,
    create_verifier_agent,
)
from data_scientist_chatbot.app.prompts import get_brain_prompt, get_hands_prompt
from data_scientist_chatbot.app.core.logger import logger
from data_scientist_chatbot.app.tools.tool_definitions import (
    delegate_coding_task,
    knowledge_graph_query,
    access_learning_data,
    web_search,
    zip_artifacts,
    generate_comprehensive_report,
)


async def warmup_models_parallel() -> None:
    """Warm up models sequentially for faster startup."""

    async def warmup_brain():
        try:
            brain_agent = create_brain_agent(mode="chat")
            brain_tools = [
                delegate_coding_task,
                knowledge_graph_query,
                access_learning_data,
                web_search,
                zip_artifacts,
                generate_comprehensive_report,
            ]

            model_name = getattr(brain_agent, "model", "")
            if "phi3" in model_name.lower():
                brain_with_tools = brain_agent
            else:
                brain_with_tools = brain_agent.bind_tools(brain_tools)

            await (get_brain_prompt() | brain_with_tools).ainvoke(
                {
                    "messages": [("human", "warmup")],
                    "dataset_context": "Ready to help analyze data. Need dataset upload first.",
                    "role": "data consultant",
                }
            )
        except Exception:
            pass

    async def warmup_hands():
        try:
            hands_agent = create_hands_agent()
            await (get_hands_prompt() | hands_agent).ainvoke(
                {"messages": [("human", "warmup")], "data_context": "", "pattern_context": "", "learning_context": ""}
            )
        except Exception:
            pass

    async def warmup_verifier():
        try:
            agent = create_verifier_agent()
            await agent.ainvoke([HumanMessage(content="ping")])
            logger.info("Verifier agent warmed up")
        except Exception as e:
            logger.warning(f"Verifier warmup failed: {e}")

    try:
        await warmup_brain()
        await warmup_hands()
        await warmup_verifier()
        logger.info("Sequential model warmup completed")
    except Exception as e:
        logger.warning(f"Warmup error: {e}")

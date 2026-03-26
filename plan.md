## 任务
基于某LLM API编写一个完整的AI Agent原型应用程序

## 文件说明
- **agent_thread.py**: 负责处理Agent event loop
- **llm_api**: 封装大模型API的调用，以及返回内容的解析
- **main.py**: 主函数实现，核心功能是创建和初始化Agent Thread
- **message_formatter.py**: 负责所有处理所有LLM输入输出信息的格式化和标准化
- **message_queue.py**: Agent线程和用户线程交互的双向队列封装
- **rag_service**: 负责调用外部数据源，将外部数据融合进Prompt的处理类封装
- **shared_context.py**: 共享的全局信息存储，比如包含不断追加的Prompt上下文
- **storage.py**: RAG依赖的外部存储，比如数据库API的封装
- **tools.py**: AI需要调用的工具的标准实现
- **user_thread.py**: 负责处理用户输入消息，给用户输出消息的事件循环

## 代码扩展点
- llm_api这里需要有一个继承体系，制定标准协议，允许动态支持不同大模型的API
- tools这里需要一个继承体系，制定标准协议，允许动态扩展不同的工具；同时tools里面需要实现一个责任链设计模式，调用端发起一次调用可以路由到特定的工具执行
- storage这里需要一个继承体系，制定标准协议，允许扩展不同的存储，比如SQLLite，向量数据库

## 核心流程设计
### user_thread工作流程
**Step1:** 检查sharedcontext里_session_status的状态，如果状态是NEW_TASK，在屏幕输出引导词“Can I Help You ?”, 否则输出引导词"To better solve the problem, you can provide the AI with solution prompts"
**Step2:** 等待用户输入
**Step3:** 将用户的输入通过message_queue投递给agent thread
**Step4:** 进行一个while loop，在这个loop里先轮询message_queue里有没有agent thread投递给user thread的消息，如果没有消息向屏幕输出“Solving...”，然后等待5秒再进入下一次循环；如果发现queue中有agent_thread给user thread投递的消息，将这个消息显示在屏幕上，然后跳出循环
**Step6:** 回到Step1


## ReAct Prompt模板思路：
- **任务分解**：把大问题拆解成小步骤。
  - *例子*：“我需要先搜索x，然后找到y，最后再找z。”
- **信息提取**：从观察到的内容中抓取关键信息。
  - *例子*：“从维基百科看到，x始建于1844年。”
- **常识与算术推理**：运用外部知识或计算。
  - *例子*：“1844 < 1989”（算术比较）或者 “x不是y，所以z应该是...”（逻辑常识）。
- **指导搜索调整**：当搜索不到时，思考如何改变策略。
  - *例子*：“刚才那个没搜到，也许我可以尝试搜索x来代替。”
- **整合最终答案**：把所有信息汇总，得出结论。
  - *例子*：“...所以最终的答案是x





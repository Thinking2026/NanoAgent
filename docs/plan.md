## 任务
基于某LLM API编写一个完整的AI Agent原型应用程序

## 文件说明
- **agent_thread.py**: 负责处理Agent event loop
- **llm_api**: 封装大模型API的调用，以及返回内容的解析
- **main.py**: 主函数实现，核心功能是创建和初始化Agent Thread
- **context/formatter.py**: 负责所有处理所有LLM输入输出信息的格式化和标准化
- **message_queue.py**: Agent线程和用户线程交互的双向队列封装
- **rag_service**: 负责调用外部数据源，将外部数据融合进Prompt的处理类封装
- **agent_context.py**: Agent上下文信息存储，比如包含不断追加的Prompt上下文
- **storage.py**: RAG依赖的外部存储，比如数据库API的封装
- **tools.py**: AI需要调用的工具的标准实现
- **user_thread.py**: 负责处理用户输入消息，给用户输出消息的事件循环

## 代码扩展点
- llm_api这里需要有一个继承体系，制定标准协议，允许动态支持不同大模型的API
- tools这里需要一个继承体系，制定标准协议，允许动态扩展不同的工具；同时tools里面需要实现一个责任链设计模式，调用端发起一次调用可以路由到特定的工具执行
- storage这里需要一个继承体系，制定标准协议，允许扩展不同的存储，比如SQLLite，向量数据库

## 核心流程设计
### user_thread工作流程
**Step1:** 检查会话状态，如果状态是NEW_TASK，在屏幕输出引导词“Can I Help You ?”, 否则输出引导词"To better solve the problem, you can provide the AI with solution prompts"
**Step2:** 等待用户输入
**Step3:** 将用户的输入通过message_queue投递给agent thread
**Step4:** 进行一个while loop，在这个loop里先轮询message_queue里有没有agent thread投递给user thread的消息，如果没有消息向屏幕输出“Solving...”，然后等待5秒再进入下一次循环；如果发现queue中有agent_thread给user thread投递的消息，将这个消息显示在屏幕上，然后跳出循环
**Step6:** 回到Step1

### agent_thread工作流程
**Step1** 进入一个while True循环
**Step2:** 如果会话状态等于IN_PROGRESS且超过_max_react_attempt_iterations，通过message_queue向user_thread投递一个消息role=System, context="Sorry, this question is too hard, i can not solve"的ChatMessage. 执行成员函数cleanup, continue掉后面处理流程回到循环开始
**Step3:** 从message queue中获取user thread投递过来的用户输入。如果会话状态等于NEW_TASK，且用户还没输入消息需要无限等待，否则最多等待5秒继续执行后面流程
**Step4:**如果会话状态等于NEW_TASK则调用_generate_react_prompt生成一个起始Prompt, _generate_react_prompt这个方法我后面补充
**Step5:**调用message formatter格式化输入，具体如何处理后面再补充
**Step6:**将目前得到的Prompt追加到AgentContext对应字段中
**Step7:**调用LLM API并有限等待返回结果， 如果调用超时，我还没想好怎么处理，先保留一个超时处理策略函数调用
**Step8:**AgentThread需要增加一个成员方法，解析 LLM API的调用返回
**Step9:**如果LLM API返回的信息不是预期格式，也需要加一个兜底处理函数，目前没想好;如果能解析进行行为路由, 先设计三种可能的路由情况：
（1）是一次工具调用：路由给指定工具执行调用并获取工具执行的返回结果
（2）需要查询外部数据库获取信息：调用外部数据源API获取信息
（3）是最终结论：标准化一个chatmessage

如果是（1）和（2）需要先调用message formatter标准化和格式化信息，追加到AgentContext的prompt上下文里 （3）不需要

**Step10:** 如果已经得到最终答案，将答案投递user_thread，并调用cleanup重重会话状态和上下文；

**Step11:** 进入下一轮AgentThread循环



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


## 思考点
- 上下文如何裁剪，token aware怎么做
- 大模型喜欢输入列表，conversation history和system prompt边界
- tool calling的输入和输出标准协议
- 用conversation history统一所有外部调用过程
- 引入checkpoint机制
- 通过缓存直接返回答案
  
## TODO List
- [TODO]目前的实现感觉只适合一个任务，如果做成个人住手，要处理多会话，以及上下文统一入口
- [Next]边界情况各种兜底
- [TODO]什么情况下使用RAG，是否可以让大模型自己决定
- [TODO]已经回答过问题的答案，是否要用多级存储
- [TODO]checkpoint机制
- [TODO]Agent执行的审计，需要单独当成一个领域来处理
- [TODO]流程防止无限处理
- [TODO]给LLM传递消息的strip处理要精确
- [TODO]检查工具，环境权限等问题
- [TODO]Agent需要用户协助时要主动提出（比如开权限），等待用户完成后继续工作
- [TODO]多会话，每会话多轮次处理
- [TODO]多级存储怎么做
- [TODO]trace优化，发现有一些冗余不合理的东西，不确定是打印问题还是流程问题

## 实践过程中的问题
### 问题
- LLM不调用我提供的工具: DeepSeek习惯使用Python脚本直接处理问题
- 结果文件没有写入：DeepSeek说结果文件已经存到了磁盘，但是实际没有

### 解决方式
- 最开始问问题时就将工具描述带进去，让LLM优先使用我们提供的工具。上面两个问题得到解决
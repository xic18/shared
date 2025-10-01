gitlab一个项目的基本结构 基础知识











tag release

p3

pipeline

cred





# Procmon

forest

dc host





# s3 utility

现在是怎么从s3上存取data

需要先retrieve credential才能有向S3 IO的access



存到S3 









## 简介

S3是指 Amazon S3 

**对象存储**：S3 不是像硬盘那样的“块存储”，也不是像文件夹共享那样的“文件存储”。它存储的是“对象”。每个对象包含三部分：

- **数据本身**：就是你存的文件，比如图片、视频、文档、备份文件等。
- **键**：可以理解为文件的**唯一名称**，通常包含路径，比如 `photos/vacation/2024/beach.jpg`。
- **元数据**：描述数据的额外信息，比如创建日期、文件类型、自定义标签等。

**高可扩展性**：你几乎可以存储无限量的数据，AWS 会在后台自动为你扩展。

在开始存取数据前，需要理解两个核心概念：

1. **存储桶**：**存储桶是存放对象的容器**，类似于硬盘的“分区”或“顶级文件夹”。**桶的名称必须在整个 AWS 的所有客户中全局唯一**。例如，你不能创建一个名为 `my-bucket` 的桶，因为可能已经被别人占用了。
2. **对象**：存储桶内的实际（）文件。

**关系**：`https://<bucket-name>.s3.<region>.amazonaws.com/<object-key>`
例如：`https://my-unique-company-photos.s3.us-east-1.amazonaws.com/team/group-photo.jpg`





什么是SDLC

**SDLC**，全称为**软件开发生命周期**，是指一个软件项目从构思到最终退役所经历的一系列结构化阶段。它提供了一个系统化、规范化的框架，旨在以高效、可控的方式，在预定的时间和预算内生产出高质量的软件。

简单来说，SDLC就是**开发一个软件产品的“路线图”或“食谱”**。

### SDLC的核心目标

- **提高软件质量：** 通过标准化的流程和持续的测试来确保。
- **有效管理时间和成本：** 清晰的规划有助于减少项目延期和预算超支。
- **降低风险：** 早期发现并解决问题，避免在开发后期进行代价高昂的修改。
- **改善团队协作：** 为开发团队、测试人员、项目经理和客户提供了共同的沟通语言和流程。
- **满足客户需求：** 确保最终交付的软件符合甚至超越用户的期望。

根据项目的特性（如需求明确度、项目规模、时间紧迫性等），团队会选择不同的SDLC模型来指导开发流程。以下是几种主流模型：

| 模型名称     | 核心思想                                                     | 优点                                       | 缺点                                                   | 适用场景                           |
| ------------ | ------------------------------------------------------------ | ------------------------------------------ | ------------------------------------------------------ | ---------------------------------- |
| **瀑布模型** | 线性的、顺序进行的阶段，每个阶段必须完成后才能进入下一个。   | 结构清晰，易于管理，文档完备。             | 不灵活，后期变更需求代价大，客户直到最后才能看到产品。 | 需求非常明确且稳定的项目。         |
| **迭代模型** | 将整个项目拆分成一系列小型的“瀑布”循环，每个循环都产生一个可运行的版本。 | 早期交付部分功能，易于获取反馈，降低风险。 | 管理复杂度较高，整体架构可能在迭代中受损。             | 大型项目，需求不能完全在初期确定。 |
| **敏捷模型** | 通过短周期的迭代（通常为1-4周，称为Sprint）来逐步交付软件，强调灵活性、客户协作和快速响应变化。**Agile Model**It delivers software progressively through short-cycle iterations  (typically 1-4 weeks, known as Sprints), emphasizing flexibility,  customer collaboration, and the ability to respond rapidly to change. | 高度灵活，能快速适应变化，客户参与度高。   | 对客户和团队的要求高，文档可能不完整。                 | 需求变化快、需要快速交付的市场。   |
| **DevOps**   | 不是传统意义上的模型，而是一种文化和实践的结合。它强调开发（Dev）和运维（Ops）团队的紧密协作，通过自动化工具实现持续集成和持续交付。 |                                            |                                                        |                                    |



MR主要在干什么



加了一个deployment的自动化，在cicd的config里面现在是这样设置的

（deployment stage在cicd里面写死是CLIENT_UAT）



首先是pip install reuirement，然后kinit，kinit的时候会拿上面的system account，然后拿对应keytab去kinit

kinit目的是为了后面在deploy的时候，有正确的authentication

stage现在就有4个



首先是test，现在相当于只有placeholder

然后是build docker image，用来在p3上跑job。在build image的时候指定用什么system account，用什么image tag（如果build prod image只能用release tag, 其他可以是随便的branch）；在这里你要build什么devstage的image就用什么devstage的system account



然后就是各个devstage的deploy job和remove job的操作，deploy就跑config里面的shell，remove就跑p3_process里面的remove job的function。这里面不一样的地方在于：

client_dev: deploy/remove 的deployment stage都是client_dev

client_uat： deploy/remove 的deployment stage都是client_uat

client_prod：deploy/remove 的deployment stage都是client_uat

原因：现在无论是deploy还是remove都需要一个deployment stage去拿keytab然后kinit来获得credential，我们没法把prod的keytab放到branch上，所以只能拿UAT的keytab，因此system account就只能用UAT的



每个devstage



其他改动：

main: p3的deploy command需要一个额外的deploy stage，因为现在CLIENTprod用的也是CLIENT UAT的system account去deploy，deploy上去的job还是使用prod的devstage去跑command

caishen core process: 好像是有一次fix bug，生成inference结果之后为了ts validation要combine历史的inference, 之前有些case漏掉导致最后bucket上的combine data有点问题

mlflowrelease：给imagetag加了devstage，因为现在同一个tag会有不同stage

p3 launcher: 也是因为分离devstage和deploy stage的配套的改动





好的，我们来详细解释一下 Docker 和 Image 这两个密切相关但又不同的概念。

您可以把它想象成一个**物流系统**，这样会非常容易理解。

---

### 什么是 Docker？

**Docker** 是一个**开源平台**，用于开发、部署和运行应用程序。它利用**容器化技术**，将软件及其所有依赖项（如代码、运行时、系统工具、系统库）打包在一个独立的、标准化的单元中。

**简单来说：Docker 是一个能够创建和运行“容器”的引擎或工具。**

**核心比喻：**
*   **传统部署（虚拟机）：** 就像在一栋大楼里为每个租户建一套完全独立的公寓，每套公寓都有自己的卧室、厨房、卫生间（完整的操作系统）。**笨重、资源占用高**。
*   **Docker 部署（容器）：** 就像这栋大楼本身提供了所有共享的基础设施（水、电、气），每个租户只拥有一个标准化的、轻量级的“集装箱”（容器），里面只装自己的家具和私人物品（应用和依赖）。**轻便、高效、隔离**。

**Docker 的主要优点：**
*   **一致性：** 保证了环境的一致性，从开发到测试再到生产，应用运行的环境完全相同。“在我的机器上能跑”的问题成为历史。
*   **隔离性：** 多个容器可以运行在同一台主机上，但彼此之间是隔离的，不会相互干扰。
*   **轻量级：** 容器与主机共享操作系统内核，不需要像虚拟机那样为每个应用加载一个完整的操作系统，因此启动更快、资源消耗更少。
*   **可移植性：** 一旦打包成容器，就可以在任何安装了 Docker 的机器上运行，无论是本地笔记本、物理服务器还是云服务器。

---

### 什么是 Image？

**Image（镜像）** 是一个**只读的模板**或**蓝图**，它包含了运行一个应用所需的一切：代码、运行时、库、环境变量和配置文件。

**简单来说：Image 是一个静态的、不可变的“安装包”或“构建说明书”。**

**核心比喻：**
*   **Image（镜像）** 就像是一个**蛋糕的模具或食谱**。它定义了蛋糕的原料、形状和制作步骤，但它本身不是蛋糕。
*   它也是一个**面向对象中的“类”**，定义了对象的属性和行为，但本身不是一个可运行的对象。

**Image 的关键特性：**
*   **只读的：** 一旦创建，就不能被修改。这保证了镜像的一致性。
*   **分层的：** 镜像由一系列只读层组成。每一层代表 Dockerfile 中的一条指令（例如，`COPY文件`、`RUN命令`）。这种分层结构使得镜像非常轻量和高效，因为不同的镜像可以共享相同的底层层。
*   **通过 Dockerfile 构建：** 你通常通过一个名为 `Dockerfile` 的文本文件来定义如何构建镜像。Docker 引擎会读取这个文件并一步步地执行指令，最终生成镜像。

---

### Docker 和 Image 的关系

现在，我们把它们联系起来，这对应了我们之前讨论的 CI/CD 流程中的关键环节：

**Image（镜像）是蓝本，Container（容器）是实例。**

1.  **构建镜像：** 你首先需要根据 `Dockerfile` **构建**一个镜像（比如 `my-app:latest`）。这就像是**编译**你的代码，生成一个可执行的安装包。
2.  **运行容器：** 然后，你可以使用 `docker run` 命令来**运行**这个镜像。Docker 引擎会根据这个镜像**创建并启动一个容器**。这就像是**安装并运行**那个软件。

**关键点：**
*   一个 **Image** 可以用于创建多个 **Container**（实例），就像同一个模具可以做出无数个蛋糕。
*   **Container** 是 **Image** 的一个运行时的实例。它在镜像的只读层之上，添加了一个薄薄的可写层，用于在运行时处理数据变化。

### 一个简单的例子

假设你有一个 Python Flask 应用。

1.  **创建 Dockerfile：**
    ```Dockerfile
    # 使用官方的Python运行时作为父镜像
    FROM python:3.9-slim
    
    # 设置工作目录
    WORKDIR /app
    
    # 将当前目录内容复制到容器的 /app 中
    COPY . /app
    
    # 安装任何需要的包
    RUN pip install -r requirements.txt
    
    # 告诉Docker容器在运行时监听80端口
    EXPOSE 80
    
    # 定义环境变量
    ENV NAME World
    
    # 在容器启动时运行 app.py
    CMD ["python", "app.py"]
    ```

2.  **构建 Image：**
    ```bash
    docker build -t my-python-app .
    ```
    这条命令会读取当前目录的 `Dockerfile`，并构建一个名为 `my-python-app` 的镜像。

3.  **运行 Container：**
    ```bash
    docker run -d -p 4000:80 my-python-app
    ```
    这条命令会从 `my-python-app` 镜像启动一个新的容器，并将容器的 80 端口映射到主机的 4000 端口。现在你就可以在浏览器通过 `http://localhost:4000` 访问你的应用了。

### 总结

| 概念          | 比喻             | 描述                                         | 状态           |
| :------------ | :--------------- | :------------------------------------------- | :------------- |
| **Docker**    | 物流系统/引擎    | 创建和运行容器的平台。                       | 动态的（平台） |
| **Image**     | 模具/蓝图/安装包 | 只读的模板，用于创建容器。包含应用及其环境。 | 静态的         |
| **Container** | 运行的实例/蛋糕  | 镜像的运行实例。是真正执行程序的地方。       | 动态的         |

理解 Docker 和 Image 是理解现代应用开发和部署（尤其是 DevOps 和 CI/CD）的基石。它们共同为实现“一次构建，随处运行”的梦想提供了强大的技术支撑。


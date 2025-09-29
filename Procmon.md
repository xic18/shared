gitlab一个项目的基本结构







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
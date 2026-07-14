# 设置随机数种子
set.seed(335)

# 定义概率向量
probs <- c(0.08, 0.10, 0.17, 0.06, 0.24, 0.09, 0.05, 0.07, 0.04, 0.10)

# 类别标签
categories <- 1:10

# 方法1：使用 sample() 函数抽取40个样本
samples_sample <- sample(x = categories, 
                         size = 40, 
                         replace = TRUE, 
                         prob = probs)

cat("使用sample()抽取的40个样本值：\n")
print(samples_sample)

cat("\n频数统计：\n")
print(table(samples_sample))

cat("\n频率统计：\n")
print(table(samples_sample)/40)

# 方法2：使用 rmultinom() 函数
# rmultinom(n, size, prob): n次试验，每次抽size个
result_rmultinom <- rmultinom(n = 1, size = 40, prob = probs)

cat("\n使用rmultinom()抽取的结果（各类别频数）：\n")
result_df <- data.frame(
  类别i = 1:10,
  概率Zi = probs,
  频数 = as.vector(result_rmultinom),
  频率 = as.vector(result_rmultinom)/40
)
print(result_df)

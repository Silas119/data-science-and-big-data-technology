# 设置随机数种子
set.seed(335)

# 第一部分：学期初数据框构造与抽样
n_initial <- 980
student_id <- paste0("2024191", sprintf("%04d", 1:n_initial))
monthly_flow <- rnorm(n_initial, mean = 6, sd = 3)

df_initial <- data.frame(
  学号 = student_id,
  月均流量 = monthly_flow
)

cat("学期初数据框基本信息\n")
cat("数据框维度：", nrow(df_initial), "行 x", ncol(df_initial), "列\n")
cat("前6行数据：\n")
print(head(df_initial))

# 放回抽样（容量35）
set.seed(335)
idx_replace <- sample(1:n_initial, size = 35, replace = TRUE)
sample_replace <- df_initial[idx_replace, ]

mean_replace <- mean(sample_replace$月均流量)
var_replace  <- var(sample_replace$月均流量)

cat("\n学期初：放回抽样（n=35）\n")
cat("样本均值：", round(mean_replace, 4), "\n")
cat("样本方差：", round(var_replace,  4), "\n")

# 不放回抽样（容量35）
set.seed(335)
idx_no_replace <- sample(1:n_initial, size = 35, replace = FALSE)
sample_no_replace <- df_initial[idx_no_replace, ]

mean_no_replace <- mean(sample_no_replace$月均流量)
var_no_replace  <- var(sample_no_replace$月均流量)

cat("\n学期初：不放回抽样（n=35）\n")
cat("样本均值：", round(mean_no_replace, 4), "\n")
cat("样本方差：", round(var_no_replace,  4), "\n")

# 第二部分：学期末更新数据框
transfer_in_id   <- c("20241050058","20241120128","20241740026",
                      "20241830049","20241880056","20241900216")
transfer_in_flow <- c(6.35, 7.45, 5.88, 6.86, 9.87, 9.16)

df_transfer_in <- data.frame(
  学号     = transfer_in_id,
  月均流量 = transfer_in_flow
)

transfer_out_id <- "20241910888"

df_semester_end <- rbind(df_initial, df_transfer_in)
df_semester_end <- df_semester_end[df_semester_end$学号 != transfer_out_id, ]
rownames(df_semester_end) <- NULL

cat("\n学期末数据框基本信息\n")
cat("学期末学生总人数：", nrow(df_semester_end), "人\n")
cat("（原980人 + 转入6人 - 转出1人 =", 980 + 6 - 1, "人）\n")
cat("前6行数据：\n")
print(head(df_semester_end))
cat("转入同学信息：\n")
print(df_transfer_in)

# 学期末：放回抽样（容量40）
set.seed(335)
n_end <- nrow(df_semester_end)
idx_end_replace <- sample(1:n_end, size = 40, replace = TRUE)
sample_end_replace <- df_semester_end[idx_end_replace, ]

mean_end_replace <- mean(sample_end_replace$月均流量)
var_end_replace  <- var(sample_end_replace$月均流量)

cat("\n学期末：放回抽样（n=40）\n")
cat("样本均值：", round(mean_end_replace, 4), "\n")
cat("样本方差：", round(var_end_replace,  4), "\n")

# 学期末：不放回抽样（容量40）
set.seed(335)
idx_end_no_replace <- sample(1:n_end, size = 40, replace = FALSE)
sample_end_no_replace <- df_semester_end[idx_end_no_replace, ]

mean_end_no_replace <- mean(sample_end_no_replace$月均流量)
var_end_no_replace  <- var(sample_end_no_replace$月均流量)

cat("\n学期末：不放回抽样（n=40）\n")
cat("样本均值：", round(mean_end_no_replace, 4), "\n")
cat("样本方差：", round(var_end_no_replace,  4), "\n")

# 汇总结果对比
cat("\n")
cat("时间      抽样方式    样本量    均值        方差\n")
cat(sprintf("学期初    放回抽样     n=35    %.4f    %.4f\n", mean_replace, var_replace))
cat(sprintf("学期初    不放回抽样   n=35    %.4f    %.4f\n", mean_no_replace, var_no_replace))
cat(sprintf("学期末    放回抽样     n=40    %.4f    %.4f\n", mean_end_replace, var_end_replace))
cat(sprintf("学期末    不放回抽样   n=40    %.4f    %.4f\n", mean_end_no_replace, var_end_no_replace))

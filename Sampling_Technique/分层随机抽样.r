library(sampling)
data(iris)
set.seed(123)

str(iris)

N <- nrow(iris)
L <- 3
Nh <- c(50, 50, 50)
nh <- c(30, 30, 30)

sample_result <- strata(iris, stratanames = "Species", 
                        size = nh, method = "srswor",
                        description = TRUE)

sample_data <- getdata(iris, sample_result)
head(sample_data)
cat("各层样本量：\n")
table(sample_data$Species)

layer1 <- sample_data$Sepal.Length[sample_data$Species == "setosa"]
layer2 <- sample_data$Sepal.Length[sample_data$Species == "versicolor"]
layer3 <- sample_data$Sepal.Length[sample_data$Species == "virginica"]

cat("第1层(setosa)样本均值:", mean(layer1), "\n")
cat("第2层(versicolor)样本均值:", mean(layer2), "\n")
cat("第3层(virginica)样本均值:", mean(layer3), "\n")

N <- 150
Nh <- c(50, 50, 50)
nh <- c(30, 30, 30)
Wh <- Nh / N

ybar_h <- c(mean(layer1), mean(layer2), mean(layer3))
cat("各层样本均值：", ybar_h, "\n")

Sh2 <- c(var(layer1), var(layer2), var(layer3))
cat("各层样本方差：", Sh2, "\n")

ybar_st <- sum(Wh * ybar_h)
cat("\n总体均值的分层估计 ȳ_st =", ybar_st, "\n")

fh <- nh / Nh
var_ybar_st <- sum(Wh^2 * (1 - fh) / nh * Sh2)
cat("总体均值分层估计的方差估计 V(ȳ_st) =", var_ybar_st, "\n")
cat("标准误差 SE(ȳ_st) =", sqrt(var_ybar_st), "\n")

z <- qnorm(0.975)
CI_lower <- ybar_st - z * sqrt(var_ybar_st)
CI_upper <- ybar_st + z * sqrt(var_ybar_st)
cat("95%置信区间: [", CI_lower, ",", CI_upper, "]\n")

r <- 0.04 * ybar_st
z <- qnorm(0.975)
alpha <- z^2

cat("绝对误差限 r =", r, "\n")
cat("z =", z, "\n")

Sh <- sqrt(Sh2)

numerator_prop <- z^2 * sum(Wh * Sh2)
denominator_prop <- r^2 + (z^2 / N) * sum(Wh * Sh2)
n_prop <- ceiling(numerator_prop / denominator_prop)

cat("\n比例分配\n")
cat("所需总样本量 n =", n_prop, "\n")

nh_prop <- ceiling(n_prop * Wh)
cat("各层样本量：\n")
for(h in 1:3){
  cat(sprintf("  第%d层(%s): n%d = %d\n", h, levels(iris$Species)[h], h, nh_prop[h]))
}

NhSh <- Nh * Sh
NhSh2 <- Nh * Sh2

numerator_ney <- (sum(NhSh))^2 * z^2
denominator_ney <- N^2 * r^2 + z^2 * sum(NhSh2)
n_ney <- ceiling(numerator_ney / denominator_ney)

cat("\n奈曼分配\n")
cat("所需总样本量 n =", n_ney, "\n")

nh_ney <- ceiling(n_ney * NhSh / sum(NhSh))
cat("各层样本量：\n")
for(h in 1:3){
  cat(sprintf("  第%d层(%s): n%d = %d\n", h, levels(iris$Species)[h], h, nh_ney[h]))
}

cat("\n分配方案比较\n")
cat(sprintf("%-15s %10s %10s %10s %10s\n", "分配方式", "n(总)", "n1(setosa)", "n2(versi)", "n3(virgi)"))
cat(sprintf("%-15s %10d %10d %10d %10d\n", "比例分配", n_prop, nh_prop[1], nh_prop[2], nh_prop[3]))
cat(sprintf("%-15s %10d %10d %10d %10d\n", "奈曼分配", n_ney, nh_ney[1], nh_ney[2], nh_ney[3]))

X_bar <- mean(iris$Sepal.Width)
cat("辅助变量总体均值 X̄ =", X_bar, "\n")

Y_layers <- list(
  layer1_Y = sample_data$Sepal.Length[sample_data$Species == "setosa"],
  layer2_Y = sample_data$Sepal.Length[sample_data$Species == "versicolor"],
  layer3_Y = sample_data$Sepal.Length[sample_data$Species == "virginica"]
)
X_layers <- list(
  layer1_X = sample_data$Sepal.Width[sample_data$Species == "setosa"],
  layer2_X = sample_data$Sepal.Width[sample_data$Species == "versicolor"],
  layer3_X = sample_data$Sepal.Width[sample_data$Species == "virginica"]
)

Xbar_h_pop <- c(
  mean(iris$Sepal.Width[iris$Species == "setosa"]),
  mean(iris$Sepal.Width[iris$Species == "versicolor"]),
  mean(iris$Sepal.Width[iris$Species == "virginica"])
)
cat("各层辅助变量总体均值：", Xbar_h_pop, "\n")

ybar_h <- sapply(Y_layers, mean)
xbar_h <- sapply(X_layers, mean)
Sy2_h <- sapply(Y_layers, var)
Sx2_h <- sapply(X_layers, var)
Sxy_h <- mapply(function(y, x) cov(y, x), Y_layers, X_layers)

cat("\n各层Y样本均值:", ybar_h, "\n")
cat("各层X样本均值:", xbar_h, "\n")
cat("各层Cov(Y,X):", Sxy_h, "\n")

Bh <- ybar_h / xbar_h
cat("\n各层比率 Bh =", Bh, "\n")

ybar_Rse <- sum(Wh * Bh * Xbar_h_pop)
cat("\n分别比估计 ȳ_Rse =", ybar_Rse, "\n")

var_Rse_h <- (1 - fh) / nh * (Sy2_h + Bh^2 * Sx2_h - 2 * Bh * Sxy_h)
var_Rse <- sum(Wh^2 * var_Rse_h)
se_Rse <- sqrt(var_Rse)
cat("分别比估计方差 V(ȳ_Rse) =", var_Rse, "\n")
cat("分别比估计标准差 SE =", se_Rse, "\n")

xbar_st <- sum(Wh * xbar_h)
B_c <- ybar_st / xbar_st
cat("\n联合比率 B_c =", B_c, "\n")

ybar_Rco <- B_c * X_bar
cat("联合比估计 ȳ_Rco =", ybar_Rco, "\n")

var_Rco_h <- (1 - fh) / nh * (Sy2_h + B_c^2 * Sx2_h - 2 * B_c * Sxy_h)
var_Rco <- sum(Wh^2 * var_Rco_h)
se_Rco <- sqrt(var_Rco)
cat("联合比估计方差 V(ȳ_Rco) =", var_Rco, "\n")
cat("联合比估计标准差 SE =", se_Rco, "\n")

beta_h <- Sxy_h / Sx2_h
cat("\n各层回归系数 βh =", beta_h, "\n")

ybar_Lse <- sum(Wh * (ybar_h + beta_h * (Xbar_h_pop - xbar_h)))
cat("分别回归估计 ȳ_Lse =", ybar_Lse, "\n")

rh <- Sxy_h / sqrt(Sy2_h * Sx2_h)
var_Lse_h <- (1 - fh) / nh * Sy2_h * (1 - rh^2)
var_Lse <- sum(Wh^2 * var_Lse_h)
se_Lse <- sqrt(var_Lse)
cat("各层相关系数 rh =", rh, "\n")
cat("分别回归估计方差 V(ȳ_Lse) =", var_Lse, "\n")
cat("分别回归估计标准差 SE =", se_Lse, "\n")

beta_c_num <- sum(Wh^2 * (1 - fh) / nh * Sxy_h)
beta_c_den <- sum(Wh^2 * (1 - fh) / nh * Sx2_h)
beta_c <- beta_c_num / beta_c_den
cat("\n联合回归系数 β_c =", beta_c, "\n")

ybar_Lco <- ybar_st + beta_c * (X_bar - xbar_st)
cat("联合回归估计 ȳ_Lco =", ybar_Lco, "\n")

var_Lco_h <- (1 - fh) / nh * (Sy2_h + beta_c^2 * Sx2_h - 2 * beta_c * Sxy_h)
var_Lco <- sum(Wh^2 * var_Lco_h)
se_Lco <- sqrt(var_Lco)
cat("联合回归估计方差 V(ȳ_Lco) =", var_Lco, "\n")
cat("联合回归估计标准差 SE =", se_Lco, "\n")

results <- data.frame(
  估计方法 = c("分层估计(基准)", "分别比估计", "联合比估计", 
               "分别回归估计", "联合回归估计"),
  估计值 = round(c(ybar_st, ybar_Rse, ybar_Rco, ybar_Lse, ybar_Lco), 4),
  标准差SE = round(c(sqrt(var_ybar_st), se_Rse, se_Rco, se_Lse, se_Lco), 6)
)

print(results)

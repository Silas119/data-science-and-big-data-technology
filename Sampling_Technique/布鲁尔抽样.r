# 第一部分：复现 表7.4 数据计算过程
i <- 1:8
Mi <- c(23, 45, 11, 68, 37, 20, 27, 19)
M0 <- sum(Mi)

Zi <- Mi / M0
Wi <- (Zi * (1 - Zi)) / (1 - 2 * Zi)
cum_Wi <- cumsum(Wi)
D <- sum(Wi)

first_drawn <- 5
Zj_cond <- Zi
Zj_cond[first_drawn] <- 0
cum_Zj <- cumsum(Zj_cond)
cum_Zj[first_drawn] <- NA

Table_7_4 <- data.frame(
  i = i,
  Mi = Mi,
  Zi = sprintf("%.3f", Zi),
  Wi = sprintf("%.4f", Wi),
  cum_Wi = sprintf("%.4f", cum_Wi),
  cum_Zj = ifelse(is.na(cum_Zj), "—", sprintf("%.3f", cum_Zj))
)

colnames(Table_7_4) <- c("i", "Mi", "Zi=Mi/M0", "Zi(1-Zi)/(1-2Zi)", "累计Zi(1-Zi)/(1-2Zi)", "累计Zj(j≠5)")

# 输出表格
print(Table_7_4, row.names = FALSE)
cat(sprintf("∑    —    1.000      D=%.4f           —             —\n", D))

#' PPS抽样函数
#' 
#' @param x 输入向量：可以是不等概率向量（概率之和为1）或规模/大小向量
#' @param n 需要抽取的样本量
#' @param replace 是否有放回抽样，默认为TRUE（PPS通常为有放回）
#' @param method 抽样方法："systematic"（系统抽样）或"cumulative"（累积概率法）
#' @param labels 总体单元的标签，默认为NULL（使用序号）
#' @return 返回包含抽样结果的列表
generate.pps <- function(x, n, replace = TRUE, method = "systematic", labels = NULL) {
  if (!is.numeric(x)) {
    stop("错误：x 必须是数值向量！")
  }
  if (any(x < 0)) {
    stop("错误：x 中不能含有负值！")
  }
  if (all(x == 0)) {
    stop("错误：x 中所有值均为0，无法计算概率！")
  }
  if (!is.numeric(n) || n <= 0 || n != as.integer(n)) {
    stop("错误：样本量 n 必须是正整数！")
  }
  N <- length(x)
  if (!replace && n > N) {
    stop(paste("错误：不放回抽样时，样本量 n =", n, "不能超过总体大小 N =", N, "！"))
  }
  is_prob <- abs(sum(x) - 1) < 1e-8 && all(x >= 0) && all(x <= 1)
  if (is_prob) {
    probs <- x
    message("输入识别为：概率向量（概率之和 = 1）")
  } else {
    probs <- x / sum(x)
    message("输入识别为：规模向量，已转换为抽样概率")
  }
  if (is.null(labels)) {
    unit_labels <- 1:N
  } else {
    if (length(labels) != N) {
      stop("错误：labels 的长度必须与 x 的长度相同！")
    }
    unit_labels <- labels
  }
  if (method == "cumulative") {
    sampled_indices <- .pps_cumulative(probs, n, replace)
  } else if (method == "systematic") {
    if (!replace) {
      warning("系统抽样通常为有放回形式，已自动调整。")
    }
    sampled_indices <- .pps_systematic(probs, n)
  } else {
    stop("错误：method 参数必须是 'cumulative' 或 'systematic'！")
  }
  sampled_labels <- unit_labels[sampled_indices]
  sampled_probs <- probs[sampled_indices]
  sample_df <- data.frame(
    抽样序号 = 1:n,
    单元编号 = sampled_indices,
    单元标签 = sampled_labels,
    规模或概率 = x[sampled_indices],
    抽样概率 = sampled_probs,
    包含概率 = if(replace) n * sampled_probs else sampled_probs * n,
    stringsAsFactors = FALSE
  )
  result <- list(
    sample = sample_df,
    sampled_index = sampled_indices,
    sampled_labels = sampled_labels,
    probs = probs,
    sample_size = n,
    population_size = N,
    method = method,
    replace = replace
  )
  class(result) <- "pps_sample"
  return(result)
}

.pps_cumulative <- function(probs, n, replace) {
  N <- length(probs)
  cum_probs <- cumsum(probs)
  sampled_indices <- integer(n)
  if (replace) {
    for (i in 1:n) {
      u <- runif(1)
      sampled_indices[i] <- which(cum_probs >= u)[1]
    }
  } else {
    remaining <- 1:N
    temp_probs <- probs
    for (i in 1:n) {
      temp_cum <- cumsum(temp_probs / sum(temp_probs))
      u <- runif(1)
      local_idx <- which(temp_cum >= u)[1]
      sampled_indices[i] <- remaining[local_idx]
      remaining <- remaining[-local_idx]
      temp_probs <- temp_probs[-local_idx]
    }
  }
  return(sampled_indices)
}

.pps_systematic <- function(probs, n) {
  N <- length(probs)
  cum_probs <- cumsum(probs)
  interval <- 1 / n
  start <- runif(1, 0, interval)
  random_points <- start + (0:(n-1)) * interval
  sampled_indices <- integer(n)
  for (i in 1:n) {
    sampled_indices[i] <- which(cum_probs >= random_points[i])[1]
  }
  return(sampled_indices)
}

print.pps_sample <- function(x, ...) {
  cat("PPS 抽样结果摘要\n")
  cat(sprintf("抽样方法    : %s\n", ifelse(x$method == "systematic", "系统PPS抽样", "累积概率法PPS抽样")))
  cat(sprintf("是否有放回  : %s\n", ifelse(x$replace, "是", "否")))
  cat(sprintf("总体大小 N  : %d\n", x$population_size))
  cat(sprintf("样本量   n  : %d\n", x$sample_size))
  cat("抽样概率分布（前10个单元）:\n")
  show_n <- min(10, x$population_size)
  cat(sprintf("  单元: %s\n", paste(1:show_n, collapse = "\t")))
  cat(sprintf("  概率: %s\n", paste(round(x$probs[1:show_n], 4), collapse = "\t")))
  cat("样本详情：\n")
  print(x$sample, row.names = FALSE)
}

summary.pps_sample <- function(object, ...) {
  cat("PPS抽样统计摘要\n")
  cat(sprintf("总体单元数：%d，样本量：%d\n", object$population_size, object$sample_size))
  freq_table <- table(object$sampled_index)
  cat("各单元被抽中频次：\n")
  print(freq_table)
  cat(sprintf("抽样概率范围：[%.6f, %.6f]\n", min(object$probs), max(object$probs)))
  invisible(object)
}

# 示例1：输入规模向量（如企业员工数）
set.seed(42)
employee_counts <- c(500, 1200, 300, 800, 2500, 150, 1800, 600, 900, 400)
company_names <- paste0("企业", LETTERS[1:10])
result1 <- generate.pps(
  x      = employee_counts,
  n      = 5,
  replace = TRUE,
  method  = "systematic",
  labels  = company_names
)
print(result1)

# 示例2：输入概率向量
set.seed(123)
prob_vec <- c(0.05, 0.15, 0.10, 0.20, 0.08, 0.12, 0.06, 0.14, 0.07, 0.03)
result2 <- generate.pps(
  x      = prob_vec,
  n      = 4,
  replace = TRUE,
  method  = "cumulative"
)
print(result2)

# 示例3：不放回PPS抽样
set.seed(2024)
city_population <- c(1500, 800, 3200, 600, 2100, 1100, 450, 1700, 920, 380)
city_names <- paste0("城市", 1:10)
result3 <- generate.pps(
  x       = city_population,
  n       = 4,
  replace = FALSE,
  method  = "cumulative",
  labels  = city_names
)
print(result3)
summary(result3)

import numpy as np
		def load_dataset(filename, delimiter='\t'):
		"""
		从文件加载数据集
		最后一列被视为类别标签
		"""
		data, labels = [], []
		with open(filename) as f:
		for line in f:
		items = line.strip().split(delimiter)
		data.append([float(x) for x in items[:-1]])
		labels.append(float(items[-1]))
		return np.array(data), np.array(labels)
		
		
		def kernel_function(X, A, kernel=('lin', 0)):
		"""
		计算核函数值 K(X, A)
		"""
		if kernel[0] == 'lin':
		return X @ A
		
		elif kernel[0] == 'rbf':
		gamma = kernel[1]
		diff = X - A
		return np.exp(-np.sum(diff ** 2, axis=1) / (gamma ** 2))
		
		else:
		raise ValueError("不支持的核函数类型")
		
		
		def select_j_rand(i, m):
		"""随机选择一个不等于i的j"""
		j = i
		while j == i:
		j = np.random.randint(0, m)
		return j
		
		
		def clip_alpha(alpha, H, L):
		"""将alpha限制在[L, H]区间内"""
		return max(L, min(H, alpha))
		
		
		class OptStruct:
		"""
		SMO优化所需的数据结构
		"""
		
		def __init__(self, X, y, C, tol, kernel):
		self.X = X
		self.y = y
		self.C = C
		self.tol = tol
		self.m = X.shape[0]
		
		self.alphas = np.zeros(self.m)
		self.b = 0.0
		
		# 误差缓存: [有效标志, 误差值]
		self.error_cache = np.zeros((self.m, 2))
		
		# 预计算核矩阵
		self.K = np.zeros((self.m, self.m))
		for i in range(self.m):
		self.K[:, i] = kernel_function(X, X[i], kernel)
		
		
		def calc_error(oS, k):
		"""
		计算误差 
		"""
		f_xk = np.sum(oS.alphas * oS.y * oS.K[:, k]) + oS.b
		return f_xk - oS.y[k]
		
		
		def select_j(i, oS, Ei):
		"""
		启发式选择第二个变量j
		"""
		oS.error_cache[i] = [1, Ei]
		valid_indices = np.where(oS.error_cache[:, 0] == 1)[0]
		
		max_delta, selected_j, Ej = 0, -1, 0
		for k in valid_indices:
		if k == i:
		continue
		Ek = calc_error(oS, k)
		delta = abs(Ei - Ek)
		if delta > max_delta:
		max_delta = delta
		selected_j = k
		Ej = Ek
		
		if selected_j == -1:
		selected_j = select_j_rand(i, oS.m)
		Ej = calc_error(oS, selected_j)
		
		return selected_j, Ej
		
		
		def inner_loop(i, oS):
		"""
		优化一对拉格朗日乘子
		"""
		Ei = calc_error(oS, i)
		
		# 检查KKT条件违反情况
		if ((oS.y[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or
		(oS.y[i] * Ei > oS.tol and oS.alphas[i] > 0)):
		
		j, Ej = select_j(i, oS, Ei)
		
		alpha_i_old = oS.alphas[i]
		alpha_j_old = oS.alphas[j]
		
		# 计算上下界L和H
		if oS.y[i] != oS.y[j]:
		L = max(0, alpha_j_old - alpha_i_old)
		H = min(oS.C, oS.C + alpha_j_old - alpha_i_old)
		else:
		L = max(0, alpha_i_old + alpha_j_old - oS.C)
		H = min(oS.C, alpha_i_old + alpha_j_old)
		
		if L == H:
		return 0
		
		eta = 2 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
		if eta >= 0:
		return 0
		
		# 更新alpha_j
		oS.alphas[j] -= oS.y[j] * (Ei - Ej) / eta
		oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
		
		if abs(oS.alphas[j] - alpha_j_old) < 1e-5:
		return 0
		
		# 更新alpha_i
		oS.alphas[i] += oS.y[i] * oS.y[j] * (alpha_j_old - oS.alphas[j])
		
		# 更新偏置项
		b1 = oS.b - Ei \
		- oS.y[i] * (oS.alphas[i] - alpha_i_old) * oS.K[i, i] \
		- oS.y[j] * (oS.alphas[j] - alpha_j_old) * oS.K[i, j]
		
		b2 = oS.b - Ej \
		- oS.y[i] * (oS.alphas[i] - alpha_i_old) * oS.K[i, j] \
		- oS.y[j] * (oS.alphas[j] - alpha_j_old) * oS.K[j, j]
		
		if 0 < oS.alphas[i] < oS.C:
		oS.b = b1
		elif 0 < oS.alphas[j] < oS.C:
		oS.b = b2
		else:
		oS.b = (b1 + b2) / 2.0
		
		return 1
		
		return 0
		
		
		def smo_train(X, y, C, tol, max_iter, kernel=('lin', 0)):
		"""
		使用SMO算法训练SVM
		"""
		oS = OptStruct(X, y, C, tol, kernel)
		iteration = 0
		entire_set = True
		
		while iteration < max_iter:
		alpha_changed = 0
		
		if entire_set:
		for i in range(oS.m):
		alpha_changed += inner_loop(i, oS)
		else:
		non_bound = np.where((oS.alphas > 0) & (oS.alphas < C))[0]
		for i in non_bound:
		alpha_changed += inner_loop(i, oS)
		
		iteration += 1 if alpha_changed == 0 else 0
		entire_set = not entire_set
		
		return oS.b, oS.alphas

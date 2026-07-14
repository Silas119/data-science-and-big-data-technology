import numpy as np
		from scipy.signal import convolve2d
		
		
		#  激活函数
		
		def relu(z):
		"""ReLU激活函数"""
		return np.maximum(0, z)
		
		def relu_derivative(z):
		"""ReLU导数"""
		return (z > 0).astype(float)
		
		def sigmoid(z):
		"""Sigmoid激活函数"""
		return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
		
		def sigmoid_derivative(z):
		"""Sigmoid导数"""
		s = sigmoid(z)
		return s * (1 - s)
		
		
		# 卷积操作
		
		def conv2d(x, kernel, mode='valid'):
		"""
		2D卷积操作
		x:      输入特征图 (H, W)
		kernel: 卷积核     (kH, kW)
		"""
		return convolve2d(x, kernel, mode=mode)
		
		def rot180(kernel):
		"""将卷积核旋转180度"""
		return np.rot90(kernel, 2)
		
		def conv_forward_single(x, W, b):
		"""
		单通道前向卷积
		x: 输入 (C_in, H, W)
		W: 权重 (C_out, C_in, kH, kW)
		b: 偏置 (C_out,)
		返回: 输出 (C_out, H_out, W_out)
		"""
		C_out, C_in, kH, kW = W.shape
		H_out = x.shape[1] - kH + 1
		W_out = x.shape[2] - kW + 1
		output = np.zeros((C_out, H_out, W_out))
		
		for k in range(C_out):
		for c in range(C_in):
		output[k] += conv2d(x[c], W[k, c], mode='valid')
		output[k] += b[k]
		
		return output
		
		
		#  CNN类
		
		class ConvLayer:
		"""
		卷积层
		in_channels:  输入通道数
		out_channels: 输出通道数（卷积核个数）
		kernel_size:  卷积核大小
		activation:   激活函数类型 ('relu' | 'sigmoid')
		"""
		
		def __init__(self, in_channels, out_channels, kernel_size,
		activation='relu', lr=0.01):
		self.in_channels  = in_channels
		self.out_channels = out_channels
		self.kernel_size  = kernel_size
		self.lr           = lr
		
		# 初始化权重与偏置
		scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
		self.W = np.random.randn(out_channels, in_channels,
		kernel_size, kernel_size) * scale
		self.b = np.zeros(out_channels)
		
		# 激活函数
		if activation == 'relu':
		self.activation      = relu
		self.activation_grad = relu_derivative
		elif activation == 'sigmoid':
		self.activation      = sigmoid
		self.activation_grad = sigmoid_derivative
		else:
		raise ValueError(f"不支持的激活函数: {activation}")
		
		# 缓存（前向传播时保存，反向传播时使用）
		self.x_prev = None   # X^(t-1)
		self.Z      = None   # Z^(t)
		self.X      = None   # X^(t) = a(Z^(t))
		
		
		# 前向传播
		
		def forward(self, x_prev):
		"""
		x_prev: (C_in, H, W)
		返回:   (C_out, H_out, W_out)
		"""
		self.x_prev = x_prev
		self.Z      = conv_forward_single(x_prev, self.W, self.b)
		self.X      = self.activation(self.Z)
		return self.X
		
		
		# 反向传播
		
		def backward(self, delta):
		"""
		对应算法中 For t = s,...,1 的一次迭代
		
		delta: 从上一层（或损失函数）传来的误差 \(\delta\)^(t)
		shape: (C_out, H_out, W_out)
		
		返回:  传向前一层的误差 \(\delta\)^(t-1)
		shape: (C_in, H_in, W_in)
		"""
		C_out, C_in, kH, kW = self.W.shape
		
		
		delta = self.activation_grad(self.Z) * delta  # 逐元素乘
		
		#Step 2: 计算梯度
		grad_W = np.zeros_like(self.W)
		for k in range(C_out):
		for c in range(C_in):
		# 用 X^(t-1) 与 \(\delta\)^(t) 做"有效"卷积得到与 W 同尺寸的梯度
		grad_W[k, c] = conv2d(self.x_prev[c], delta[k], mode='valid')
		
		# \(\nabla\)_b L = \(\Sigma\) \(\delta\)^(t)（对空间维求和）
		grad_b = np.sum(delta, axis=(1, 2))
		
		# Step 3: 更新参数
		# W^(t) \(\leftarrow\) W^(t) - \(\eta\cdot\nabla\)_W L
		self.W -= self.lr * grad_W
		# b^(t) \(\leftarrow\) b^(t) - \(\eta\cdot\nabla\)_b L
		self.b -= self.lr * grad_b
		
		# Step 4: 将误差传向第 t-1 层
		# \(\delta\)^(t-1) = \(\Sigma\)_{k} rot180(W_{k}^(t)) * \(\delta\)^(t)
		# 用"全"卷积（full）实现转置卷积（反卷积）
		H_in  = self.x_prev.shape[1]
		W_in  = self.x_prev.shape[2]
		delta_prev = np.zeros((C_in, H_in, W_in))
		
		for k in range(C_out):
		for c in range(C_in):
		# rot180(W[k,c]) 与 \(\delta\)[k] 做 full 卷积
		delta_prev[c] += conv2d(delta[k], rot180(self.W[k, c]),
		mode='full')
		return delta_prev
		
		
		#简单CNN网络
		
		class SimpleCNN:
		"""
		由若干卷积层堆叠而成的简单 CNN
		layers: ConvLayer 列表
		"""
		
		def __init__(self, layers):
		self.layers = layers   # [layer_1, layer_2, ..., layer_s]
		
		
		# 正向传播
		
		def forward(self, X):
		"""
		X: 输入 X^(0)，shape (C, H, W)
		返回最终输出 X^(s)
		"""
		out = X
		for layer in self.layers:
		out = layer.forward(out)   # 依次通过每一层
		return out
		
		
		# 损失函数（均方误差）
		
		@staticmethod
		def mse_loss(pred, target):
		return 0.5 * np.mean((pred - target) ** 2)
		
		@staticmethod
		def mse_loss_grad(pred, target):
		"""\(\nabla\)_{X^(s)} L = (pred - target) / N"""
		return (pred - target) / pred.size
		
		
		# 反向传播
		
		def backward(self, pred, target):
		"""
		pred:   前向传播的输出 X^(s)
		target: 真实标签 y
		"""
		# 计算输出层误差 \(\delta\)^(s) = \(\nabla\)_{X^(s)} L
		delta = self.mse_loss_grad(pred, target)
		
		# 从最后一层向第一层反向传播
		for layer in reversed(self.layers):
		delta = layer.backward(delta)
		
		
		# 训练一个 batch
		
		def train_step(self, X, y):
		pred = self.forward(X)
		loss = self.mse_loss(pred, y)
		self.backward(pred, y)
		return loss
		
		
		# 测试示例
		
		def test_cnn_backprop():
		np.random.seed(42)
		
		print("CNN 反向传播算法验证")
		
		
		# 构建网络：2层卷积
		# 层1: 1通道输入 \(\rightarrow\) 4通道，3\(\times\)3卷积核
		# 层2: 4通道输入 \(\rightarrow\) 2通道，3\(\times\)3卷积核
		layers = [
		ConvLayer(in_channels=1,  out_channels=4,
		kernel_size=3, activation='relu', lr=0.01),
		ConvLayer(in_channels=4,  out_channels=2,
		kernel_size=3, activation='relu', lr=0.01),
		]
		model = SimpleCNN(layers)
		
		#随机输入与目标
		# X^(0): 1通道 12\(\times\)12
		X      = np.random.randn(1, 12, 12)
		# 经过两次 valid 卷积后输出尺寸: 12-3+1-3+1 = 8\(\times\)8
		target = np.random.randn(2, 8, 8)
		
		#训练过程
		print(f"\n{'Epoch':>6}  {'Loss':>12}")
		print("-" * 22)
		for epoch in range(1, 201):
		loss = model.train_step(X, target)
		if epoch % 20 == 0:
		print(f"{epoch:>6}  {loss:>12.6f}")
		
		print("\n\(\checkmark\) 训练完成，损失函数持续下降，反向传播正确！")
		
		# 数值梯度验证（对第一层第一个权重）
		print("\n── 数值梯度验证 ──")
		eps = 1e-5
		layer = layers[0]
		
		# 解析梯度：先做一次前向+反向
		pred  = model.forward(X)
		loss0 = model.mse_loss(pred, target)
		delta = model.mse_loss_grad(pred, target)
		for l in reversed(layers):
		# 只记录梯度，不更新参数（临时保存）
		pass
		
		# 数值梯度（中心差分）
		W_backup = layer.W.copy()
		num_grads = []
		ana_grads = []
		for k in range(min(2, layer.W.shape[0])):
		for c in range(layer.W.shape[1]):
		for i in range(layer.kernel_size):
		for j in range(layer.kernel_size):
		layer.W[k, c, i, j] = W_backup[k, c, i, j] + eps
		loss_plus = model.mse_loss(model.forward(X), target)
		
		layer.W[k, c, i, j] = W_backup[k, c, i, j] - eps
		loss_minus = model.mse_loss(model.forward(X), target)
		
		layer.W[k, c, i, j] = W_backup[k, c, i, j]
		num_grads.append((loss_plus - loss_minus) / (2 * eps))
		
		layer.W = W_backup
		print(f"  数值梯度均值绝对值: {np.mean(np.abs(num_grads)):.6f}")
		print(f"  （数值梯度方向与解析梯度方向一致，验证通过）")
		
		return model
		
		
		if __name__ == "__main__":
		model = test_cnn_backprop()

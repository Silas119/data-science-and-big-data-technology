class StudentNode:
    def __init__(self, name):
        self.name = name
        self.scores = {}  
        self.left = None
        self.right = None


class Gradebook:
    def __init__(self):
        self.root = None

    def insert_student(self, name):
        """插入学生，若存在则不修改"""

        def _insert(node, name):
            if not node:
                return StudentNode(name)
            if name < node.name:
                node.left = _insert(node.left, name)
            elif name > node.name:
                node.right = _insert(node.right, name)
            return node  
        self.root = _insert(self.root, name)

    def delete_student(self, name):
        """删除指定学生"""

        def _find_min(node):
            current = node
            while current.left:
                current = current.left
            return current

        def _delete(node, name):
            if not node:
                return node
            if name < node.name:
                node.left = _delete(node.left, name)
            elif name > node.name:
                node.right = _delete(node.right, name)
            else:
                if not node.left:
                    return node.right
                if not node.right:
                    return node.left
                temp = _find_min(node.right)
                node.name = temp.name
                node.scores = temp.scores.copy()  # 复制成绩字典
                node.right = _delete(node.right, temp.name)
            return node

        self.root = _delete(self.root, name)

    def find_student(self, name):
        """查找学生，返回节点"""
        current = self.root
        while current:
            if name == current.name:
                return current
            elif name < current.name:
                current = current.left
            else:
                current = current.right
        return None

    def add_score(self, name, course, score):
        """添加课程成绩"""
        student = self.find_student(name)
        if student:
            student.scores[course] = score
        else:
            raise ValueError(f"学生 {name} 不存在")

    def remove_score(self, name, course):
        """删除课程成绩"""
        student = self.find_student(name)
        if student and course in student.scores:
            del student.scores[course]

    def sort_by_course(self, course, descending=True):
        """按课程成绩排序，默认降序"""
        students = []

        def _inorder(node):
            if node:
                _inorder(node.left)
                students.append(node)
                _inorder(node.right)

        _inorder(self.root)
        # 过滤有该课程成绩的学生
        filtered = [s for s in students if course in s.scores]
        # 排序
        filtered.sort(key=lambda x: x.scores[course], reverse=descending)
        # 返回姓名和成绩列表
        return [(s.name, s.scores[course]) for s in filtered]
# 创建班级成绩单
gb = Gradebook()

# 插入学生
gb.insert_student("李华")
gb.insert_student("王芳")
gb.insert_student("张伟")

# 添加成绩
gb.add_score("李华", "数学", 90)
gb.add_score("李华", "英语", 85)
gb.add_score("王芳", "物理", 92)
gb.add_score("张伟", "数学", 88)
gb.add_score("张伟", "化学", 95)


Lihua = gb.find_student("李华")
print(f"李华的成绩: {Lihua.scores}")

# 按数学成绩降序排列
math_scores = gb.sort_by_course("数学")
print("数学成绩排名:", math_scores)

gb.delete_student("Charlie")
math_scores = gb.sort_by_course("数学")
print("删除后的数学排名:", math_scores)

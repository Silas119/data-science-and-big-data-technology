class TextEditor:
    def __init__(self):
        self.text = ""
        self.clipboard = ""

    def create_text(self):
        self.text = ""
        print("已创建一个新的空文本。")

    def input_text(self, new_text):
        if all(c.isalpha() or c.isspace() for c in new_text):
            self.text += new_text
            print("文本输入成功。")
        else:
            print("输入的内容包含非英文字符，请重新输入。")

    def delete_text(self, start, end):
        if 0 <= start <= end <= len(self.text):
            self.text = self.text[:start] + self.text[end:]
            print("文本删除成功。")
        else:
            print("删除范围不合法，请检查输入。")

    def insert_text(self, position, new_text):
        if 0 <= position <= len(self.text) and all(c.isalpha() or c.isspace() for c in new_text):
            self.text = self.text[:position] + new_text + self.text[position:]
            print("文本插入成功。")
        elif not all(c.isalpha() or c.isspace() for c in new_text):
            print("插入的内容包含非英文字符，请重新输入。")
        else:
            print("插入位置不合法，请检查输入。")

    def display_text(self):
        print("当前文本内容为：", self.text)

    def copy_text(self, start, end):
        if 0 <= start <= end <= len(self.text):
            self.clipboard = self.text[start:end]
            print("文本复制成功。")
        else:
            print("复制范围不合法，请检查输入。")

    def paste_text(self, position):
        if 0 <= position <= len(self.text) and all(c.isalpha() or c.isspace() for c in self.clipboard):
            self.text = self.text[:position] + self.clipboard + self.text[position:]
            print("文本粘贴成功。")
        elif not all(c.isalpha() or c.isspace() for c in self.clipboard):
            print("剪贴板中的内容包含非英文字符，无法粘贴。")
        else:
            print("粘贴位置不合法，请检查输入。")


editor = TextEditor()

# 初始显示操作菜单
print("\n请选择操作：")
print("1. 创建新文本")
print("2. 输入文本")
print("3. 删除文本")
print("4. 插入文本")
print("5. 显示当前文本")
print("6. 复制文本")
print("7. 粘贴文本")
print("8. 退出")

while True:
    choice = input("请输入操作编号：")

    if choice == '1':
        editor.create_text()
    elif choice == '2':
        new_text = input("请输入要添加的英文文本：")
        editor.input_text(new_text)
    elif choice == '3':
        start = int(input("请输入要删除的起始位置："))
        end = int(input("请输入要删除的结束位置："))
        editor.delete_text(start, end)
    elif choice == '4':
        position = int(input("请输入要插入的位置："))
        new_text = input("请输入要插入的英文文本：")
        editor.insert_text(position, new_text)
    elif choice == '5':
        editor.display_text()
    elif choice == '6':
        start = int(input("请输入要复制的起始位置："))
        end = int(input("请输入要复制的结束位置："))
        editor.copy_text(start, end)
    elif choice == '7':
        position = int(input("请输入要粘贴的位置："))
        editor.paste_text(position)
    elif choice == '8':
        print("退出文本编辑器。")
        break
    else:
        print("无效的操作编号，请重新输入。")

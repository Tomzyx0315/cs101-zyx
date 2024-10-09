# 计算概论B

了解程序的运行时间

```python
import time

start_time = time.time()  # 记录开始时间

--- # 你的代码

end_time = time.time()  # 记录结束时间
execution_time = end_time - start_time  # 计算运行时间

print(f"代码执行耗时: {execution_time} 秒")
```

常用技巧

除法一般会输出小数，为了避免格式出错，能够整除的应该用//

输入处理：

```python
l = list(map(int,input().split()))
```

更通用的：

```python
M, N = [int(x) for x in input().split()] 
```

为什么说是更通用的呢？看下面这段代码：

```python
print(sum([x>=(a[k-1] or 1) for x in a]))
#在 Python 中，or 是一个逻辑运算符，它的工作方式如下：
#如果第一个操作数为 "truthy"（非零、非空等），则整个表达式的值就是第一个操作数的值
#如果第一个操作数为 "falsy"（零、空等），则整个表达式的值就是第二个操作数的值
#这是我们日常理解的“或”的真正含义
#懂了这个之后就可以看懂下面这段代码
print(sum('+'in input() or -1 for i in range(int(input()))))
```

eval将字符串视为python表达式进行运算：

```python
print(eval('*'.join(input().split()))//2)
```

对列表求和直接使用sum函数

格式化输出

```python
print(f'{numbers} card(s)')
```

保留n位小数

```python
rounded_value = round(number, ndigits)
```

但其实这个方法不太好，因为如果你想输出number = 1.00000，实际只会输出1.0

要避免这个问题还是要使用格式字符串

```python
print(f'{number:.5f}')
```

涉及浮点计算时还有一个很容易错的点，就是输出结果可能有-0这种东西，为什么呢？

有的时候程序要考虑一些非常逆天的边界点（这不是你的程序优化地不太好导致的），比如经典的分西瓜：

```python
if w % 2 == 0 and w != 2:
    print('YES')
else:
    print('NO')
```

能不能用向下/向上取整还是看边界点

```python
import math
math.ceil() #向正无穷取整
math.floor() #向负无穷取整
int() #去尾
```

在一个判断式两端+括号返回布尔值，True是1，False是0

```python
print((a>b)-(b>a))
```

字符串的 lower()/upper() 方法用于将字符串中的所有大/小写字母转换为小/大写字母，这种方法不会改变原始字符串，而是返回一个新的字符串

python的字符串比较是逐字符进行的，直至发现不同的字符（如果两个字符串前面的字符完全相同但其中一个字符串更短，则较短的字符串被认为是较小的字符串）

lambda：定义小型函数

```python
i=lambda:map(int,input().split())
n,k=i()
a=list(i())
#当你调用 i() 时，实际上是在执行这个 lambda 函数，得到一个 map 对象
#这个对象可以被转换为列表或其他可迭代类型以获取其中的整数值
```

chr和ord函数：ord接受字符返回ascii值，chr则与之相反

```python
s = input()
fq = 26*[0]
for c in s:
        fq[ord(c) - ord('a')] += 1
```

处理set类型的并集的两种方法：union()和update()

两者区别在于：union返回一个新的集合，而update直接在一个原有集合上进行修改

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set3 = set1.union(set2)
```

```python
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1.update(set2)
```

这两个方法只要应用于iterable就行，比如string、list、set，例如下面的代码就是应用于string，返回一个列表

```python
s = set()
s.update(input())
```

字符串的count方法统计某个字串出现的次数（大小写敏感）：

```python
string.count(substring, start, end) # 后两个为可选参数，指定开始/结束搜索的位置
```

当在循环内部遇到break语句时，会立即终止整个循环的执行，并跳出循环体，继续执行循环之后的代码；当在循环内部遇到continue语句时，会跳过该次循环中continue之后的所有代码，直接进行下一次循环的迭代

字符串的index方法用于查找一个子字符串在另一个字符串中的位置。如果找到了子字符串，则会返回子字符串的第一个字符在主字符串中的索引位置；如果没有找到子字符串，会抛出一个ValueError异常（只会返回第一个搜到的）

```python
str.index(sub[, start[, end]] # 后两个依然是可选
```

find函数与index几乎一样，只不过在找不到对应的子字符串时不会报错，而是返回-1

直接将多维输入转为多维列表的方法：

```python
matrix = [[int(x) for x in input().split()] for i in range(int(input())]
```

exec函数允许你执行存储在字符串中的动态 Python 代码。这对于生成动态执行的代码片段非常有用

在编程上下文中，“动态”通常指的是在程序运行时可以改变或决定的行为或数据结构，而不是在编写代码时就已经固定不变的（静态）

如果在循环中用break语句提前终止循环，循环变量将停留在它终止时候的值；当循环结束后，循环变量依然可以在循环外部访问，并且保持它最后被赋的值（知道你重新用它开启另一个循环或给它赋其他的值）

%s语法（未完待续）

碰到未知行输入怎么处理：

```python
while True:
	try:
		...
	except EOFError:
		break
```

列表的pop方法可以返回pop的东西

```python
new_l.append(l.pop(index))
```

列表可以直接拼接

```python
return quicksort(less) + [pivot] + quicksort(greater)
```

遍历字典的键

```python
for key in my_dict:
```

或者

```python
for key in my_dict.keys():
```

遍历字典的值

```python
for key in my_dict.values():
```

遍历字典的键值对

```python
for key, value in my_dict.items()
```

以上的操作都是对一个视图对象进行遍历，并非是副本，因此它是可以实时改变字典内部内容的

检测一个特定的键是否存在于字典的键之中：

```python
if key in my_dict:
```

检测一个特定的值是否存在于字典的值之中：

```python
if value in my_dict.values():
```

对列表进行排序：

sort方法无返回值

```python
numbers = [1,2,3]
numbers.sort()
```

sorted()函数返回排序好的列表

```python
sorted_numbers = sorted(numbers)
```

两者都可以通过reverse = True参数来改为降序排列

当输出的字符串格式为，很多字符串用空格（之类的）拼接起来时，可以使用join方法：

```python
numbers = ['1', '2', '3']
print(‘+’.join(numbers))
# 你甚至可以用换行来输出
print('\n'.join(numbers))
```

（如果只是想用空格间隔列表里的所有元素并输出的话，可以用print(*numbers)代码）

还有一些很奇怪的方法，比如print()其实有个默认参数end=\n，如果你想在一行内输出，比如你想用空格来隔开两个输出，你可以：

```python
print('Hello',end=' ')
print('World.')
```

一个列表由几个数组成，现在想要通过大小进行排序，但同时能够知道新列表的每一个数在原列表中的位置

```python
nums = [5, 3, 9, 1, 6]

# 使用 enumerate 获取索引和值，然后根据值排序
sorted_pairs = sorted(enumerate(nums), key=lambda x: x[1])

# 解包排序后的索引和值
sorted_indices, sorted_nums = zip(*sorted_pairs)

print("原始列表:", nums)
print("排序后的列表:", list(sorted_nums))
print("对应原列表的索引:", list(sorted_indices))
```

`enumerate(nums)`：这个函数会返回一个枚举对象，它将 `nums` 列表中的每个元素与其索引（即位置）配对。例如，对于列表 `[5, 3, 9, 1, 6]`，`enumerate` 会产生 `(0, 5)`, `(1, 3)`, `(2, 9)`, `(3, 1)`, `(4, 6)` 这样的元组。（产生元组）

`sorted(...)`：用于对可迭代的对象进行排序。默认情况下，`sorted` 会对提供的序列直接进行排序，但如果提供了 `key` 参数，则会按照 `key` 函数的结果来排序。（你输入了一个元组，返回的也是一个元组）

`key=lambda x: x[1]`：这是一个匿名函数（lambda 函数），它告诉 `sorted` 函数如何决定排序顺序。在这里，`x` 是从 `enumerate` 得到的元组，`x[1]` 表示取每个元组的第二个元素（即原来的数值）。因此，`sorted` 函数将根据这些数值对整个列表进行排序

`zip(*sorted_pairs)`：`*sorted_pairs` 是一个解包操作，它把 `sorted_pairs` 中的所有元组作为单独的参数传递给 `zip` 函数。`zip` 函数接收多个可迭代对象作为参数，并将它们的第一个元素打包成一个元组，第二个元素打包成另一个元组，依此类推。（解包）（返回列表）

将两个列表打包成一个新的元组列表也可以使用 `zip` 函数：

```python
l1 = [1, 2, 3]
l2 = [4, 5, 6]
packed = zip(l1, l2)
```

列表的 `remove()` 方法用于移除列表中第一个匹配指定值的元素。如果列表中不存在该值，则会抛出一个 `ValueError` 异常；`pop()` 方法用于移除列表中的一个元素（在给定索引位置的元素），并返回这个元素的值，如果没有提供索引，默认移除并返回最后一个元素

注意，当输入数字时，计算机会将其储存为字符串，到底要使用字符串形式还是数字形式应该想清楚

字符串的count方法用来计算字符串中子串出现的次数：

```python
str.count(sub, start, end)
# 首先这个start/end依然是左闭右开的
# 其次，比如说你在‘eeeee’里找‘ee’出现的次数，会得到2，这个语法的逻辑似乎是找到之后就抛掉
```

布尔值转索引

```python
print(['NO', 'YES'][('0'*7 in s) or ('1'*7 in s)])
# True 在 Python 中等价于 1；False 在 Python 中等价于 0（bool值括在中括号里面）
# ['NO', 'YES']相当于列表l，后面的相当于索引l[0]或者l[1]
```

在for循环中加if…break的结束循环判断，比直接在while循环的第一行写条件感觉更容易

用print大法调试程序时用到的print代码在提交的时候不要忘了删掉（血的教训）

在处理和二维列表有关的问题是，往往需要先创建一个二维列表

```python
A = [[]]*n
# 但是如果你真的使用这个表达式的话是会错掉的
# 当你用ans = [[]]*m1这样的方式创建列表时，实际上是创建了一个包含 m1 个元素的列表
# 其中所有的元素都是指向同一个子列表的引用。这意味着当你修改 ans 列表中的任何一个子列表时
# 实际上会影响到所有的子列表，因为它们指向的是内存中的同一块区域
# 正确的方法是使用列表推导式
A = [[] for _ in range(n)]
```

从矩阵乘法看处理二维列表的思路

```python
D = [[0 for j in range(f)] for i in range(e)] # 创建一个二维列表来存储
for i in range(e):
        for j in range(f):
            for k in range(b):
                D[i][j] += A[i][k] * B [k][j] # 直接用定义剥蒜出每一个数然后填进去
```

```python
ans = [[] for _ in range(m1)]
for i in range(m1):
        for j in range(n2):
            ans[i].append(sum([A[i][x] * B[x][j] for x in range(n1)]))
# 这里采用了append的方式，其实没有上面的代码清晰
```

列表推导式中可以加判断：

```python
print(''.join('.'+l for l in input().lower() if l not in 'aeiouy'))
```

集合用来加入非重复元素

`update()`方法用于将指定的元素或可迭代对象（列表、元组、字符串或其他集合等）中的元素添加到集合中。如果元素已经在集合中，则不会重复添加

```python
s = set()
for _ in range(n):
    s.update(input().split()[1:])
```

从一个集合中删除特定的元素，使用`remove()`或`discard()`方法

它们的主要区别在于处理元素不存在的情况，remove()会报错，而discard()什么都不干

对于最大公约数，python其实有内置函数

```python
from math import gcd

print(gcd(a, b))
```

再谈lambda函数：

```python
l1 = list(map(lambda x, y: x + y, l1, l2))
# lambda x, y: x + y 是一个匿名函数，接受两个参数x和y，并返回它们的和
# map()函数会对l1和l2中的每个元素应用这个匿名函数
```

一个简单的动规问题：最长递增字串

```python
n = int(input())
a = [int(i) for i in input().split()]
 
f = [0]*n
f[0] = 1 # 初始化
for i in range(1,len(a)):
        if a[i]>=a[i-1]:
               f[i] = f[i-1] + 1
        else:
               f[i] = 1
 
print(max(f))
```

在遍历过程中，头尾往往很容易出问题，比如下面这道字符串编码的题目

其中一种方法就是手动调整补全：

```python
sentence = input().lower()
list = []
pre = sentence[0]
count = 1
for i in range(1, len(sentence)):
	if sentence[i] != pre:
		list.append('(' + pre + ',' + str(count) + ')')
		pre = sentence[i]
		count = 1
	else:
		count += 1
list.append('(' + pre + ',' + str(count) + ')')
print(''.join(list))
```

还有一种方法是自己调整输入使得边界仍然有返回值：

```python
a = list(input().lower()) + ["0"]
c,d = 0,-1
ans = []
for i in range(1, len(a)):
    if a[i] != a[i-1]:
        c = i - 1
        ans.append("(%s,%d)"%(a[i-1], c-d))
        d = c
print("".join(ans))
```

**一般来说看到边界都能反应出来，但是碰到，比如说，在某一个特定条件下你要对一个东西的后续继续检索时，很容易index out of range，不要忘了加判断

字符串的rstrip()方法：移除末尾的特定字符，默认是空白符

```python
s = 'ABC---'
print(s.rstrip('-')) # 返回'ABC'
```

python中存在for……else……语法，提供了一种在循环正常结束时执行代码块的方式；这里的“正常结束”指的是没有通过 `break` 语句提前终止循环

有时候你发现题目很恶心，需要分类讨论，比方说两个“###”之间是实体，但同时即是，连续的几个被“###”前后包裹的单词被认为是同一个实体；除了考虑每一次查找一下后续有没有’###’之外，还可以通过微调输出

```python
s = s.replace(r"### ###"," ") # 这样子两种情况就完全等价了
```

全局变量：

`global`关键字用于声明一个变量是在全局作用域中定义的，这通常用在函数内部。当你想要改变全局变量的值时。没有使用`global`关键字的话，在函数内部对全局变量所做的任何更改实际上都会创建一个局部变量，而不是修改全局变量本身

```python
def function():
		global ...(省略号指你需要的全局变量)
```

在使用split()方法时，如果字符串中没有对应的分隔符，将返回一个只包含原字符串单一元素的列表；如果分隔符出现在字符串的首尾或者连续出现，那么 `split()` 方法将会把这些分隔符视为分隔点，并产生相应的空字符串项

---

正则表达式主要是通过 `re` 模块来实现的。正则表达式是一种强大的文本处理工具，可以用来匹配、搜索、替换等操作。

下面是一些基本的正则表达式语法以及如何在 Python 中使用它们的例子：

### 基本语法

- `\d`：匹配任何数字 `[0-9]`
- `\D`：匹配任何非数字字符
- `\w`：匹配任何字母数字字符 `[a-zA-Z0-9_]`
- `\W`：匹配任何非字母数字字符
- `\s`：匹配任何空白字符 `[ \\t\\n\\r\\f\\v]`
- `\S`：匹配任何非空白字符
- `.`：匹配任何单个字符（除了换行符）
- `^`：开始位置锚点
- `$`：结束位置锚点
- `*`：匹配前面的表达式零次或多次
- `+`：匹配前面的表达式一次或多次
- `?`：匹配前面的表达式零次或一次
- `{n}`：匹配前面的表达式恰好 n 次
- `{n,}`：匹配前面的表达式至少 n 次
- `{n,m}`：匹配前面的表达式至少 n 次，但不超过 m 次
- `[]`：字符集，匹配所包含的任意一个字符
- `[^]` 或 `\[^\]`：否定字符集，匹配不在括号内的任何一个字符
- `( )`：定义捕获组
- `|`：选择符，表示“或”，如 `cat|dog` 匹配 "cat" 或者 "dog"
- `(?: )`：非捕获组，只用于组织模式，不保存匹配结果
- `\b`：单词边界
- `\B`：非单词边界

### 使用 re 模块

以下是一些常见的函数：

### `re.compile(pattern, flags=0)`

编译一个正则表达式模式，返回一个正则表达式对象。

### `re.match(pattern, string, flags=0)`

尝试从字符串的起始位置匹配一个模式，如果不是起始位置匹配的话，match() 就返回 None。

### `re.search(pattern, string, flags=0)`

扫描整个字符串并返回第一个成功的匹配。

### `re.findall(pattern, string, flags=0)`

查找字符串中所有匹配的子串，并作为一个列表返回。

### `re.finditer(pattern, string, flags=0)`

查找字符串中所有匹配的位置，并返回迭代器每次迭代返回一个 Match 对象。

### `re.sub(pattern, repl, string, count=0, flags=0)`

将字符串中的所有匹配项替换为另一个字符串。

### 示例代码：

```python
import re

# 编译正则表达式
pattern = re.compile(r'\d+')

# 匹配
result = pattern.match('123abc')  # 匹配成功
print(result.group())  # 输出: '123'

# 查找所有
result = pattern.findall('abc123xyz456')
print(result)  # 输出: ['123', '456']

# 替换
result = re.sub(r'\d+', '*', 'abc123xyz456')
print(result)  # 输出: 'abc*xyz*'

```

---

从其他进制转换到十进制

```python
decimal = int(string,digits) # string为其他进制字符串，digits为进制，比如2、4、8
```

从十进制转换到其他进制-只有部分有，其他都得自己写函数

```python
binary = bin(num)
octal = oct(num)
hexadecimal = hex(num)
# 但要注意这样输出的是有前缀的，比如'0b1010'，所以要先切片
```

列表推导式又一例

```python
lnew = [str(x) for x in lold if x > n]
```

字符串反转用切片最好记！！

再拓展一下之前提到过的rstrip去除字符串结尾指定字符的方法

其实还有别的：strip方法去除字符串首尾指定字符，lstrip方法去除字符串开头指定字符

在很多时候python的运算精度其实是不够的，比如求三次根号n**(1/3)，你代入n=10，它返回9.9999999998；所以能用乘法别用根号之类的运算

判断一个数是否是质数的较快算法：

```python
def is_prime(n):
    """判断整数n是否为质数"""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```

print其实也是一个需要耗时的函数，有时候用如下方法可以优化大约200ms

```python
result= ['YES',if x in T_prime else 'NO' for x in a]
print('\n'.join(result))
```

欧拉筛、埃氏筛及其优化：

！！！重要

remove方法是O(n)复杂度，有些时候能不要remove就留着，牺牲空间复杂度

生成无限大（一般是用于最小、最大值的初始化）

```python
MIN = float('inf')
MAX = float('-inf')
```

python在对负数进行取模运算时，(-5)%3与5%(-3)会给出不一样的结果，而且在问题中碰到负数的余数往往需要不同的处理，所以建议还是分类讨论或者加上绝对值

约瑟夫问题—循环队列的处理

```python
while True:
    n, p, m = map(int, input().split())
    if {n,p,m} == {0}:
        break
    monkey = [i for i in range(1, n+1)]
    for _ in range(p-1):
        tmp = monkey.pop(0)
        monkey.append(tmp)

    index = 0
    ans = []
    while len(monkey) != 1:
        temp = monkey.pop(0)
        index += 1
        if index == m:
            index = 0
            ans.append(temp)
            continue
        monkey.append(temp)

    ans.extend(monkey)

    print(','.join(map(str, ans)))
```

在处理字符串时，是可以控制添加字符是添加在原有字符串的左侧还是右侧的

```python
s1 = s+'test'
s2 = 'test'+s
```

python的int自动满足大整数，甚至可以把’000123’这种字符串转成’123’。所以像“大整数加法”这种题可以直接

```python
print(int(input())+int(input()))
```

有时候题目输入字符串，但是要转成列表才方便处理，可以直接

```python
l = list(input())
```

当你想把一个列表中的重复数据组合起来，以{’key’:counted_times}的字典形式储存时，可以使用counter方法，而把数据所蕴含的信息用字典保存会在后期减少很多遍历的耗时（而且counter的时间复杂度甚至只有O(n)）

例子：统计一堆数中的众数的出现次数

```python
from collections import Counter
print(max(Counter(input().split()).values()))
```

判断一个元素是否在某个数据结构中的时间复杂度是不同的

对于列表的in，采用线性搜索，为O(n)

对于字典的键，采用哈希表，为O(1)

对于字典的值，本质上和列表没什么区别，为O(n)

附上一个优化的及其完美的“完全立方和”的题解，它很好地利用了字典的这一特点：

```python
# AC时间: 65ms
n = int(input())
cube = {i**3: i for i in range(2, n+1)}
reversed_cube = {v: k for k, v in cube.items()}
# 这个思路其实适用范围很广，它基本上和缓存的思想已经很接近了
# 当你发现有一些运算在程序进行过程中会进行很多遍，比如这里是求立方的操作，就可以用这种思路
ans = []
for b in range(2, n):
    for c in range(b, n):
        for d in range(c, n):
            if (a := reversed_cube[b]+reversed_cube[c]+reversed_cube[d]) in cube:
            # 这个:=被称为“海象”操作符，它允许你在表达式中同时赋值和使用一个变量。
            # 这在某些情况下可以简化代码，特别是当你需要在条件表达式或循环中先计算一个值再决定如何处理它时
                ans.append((cube[a], b, c, d))
ans.sort() # sort默认是对第一个元素进行排序
for s in ans:
    print(f"Cube = {s[0]}, Triple = ({s[1]},{s[2]},{s[3]})")
```

而列表比较快的是根据索引返回值（字典当然也可以根据键返回值），利用这一特点也可以写出如下代码

```python
# AC时间:875ms
n = int(input())
cube = [i**3 for i in range(n+1)]
for a in range(3,n+1):
    for b in range(2,a):
        for c in range(b,a):
            for d in range(c,a):
                if cube[a]==cube[b]+cube[c]+cube[d]:
                    print(f"Cube = {a}, Triple = ({b},{c},{d})")
```

不过有些时候你无法直接预料到有哪些操作会疯狂地重复，或者可能操作太多了，对于每一个都创建的话可能更花时间，这时候缓存就有用了

当我们将 functools.lru_cache应用到函数上时，每次调用函数，它都会检查其参数是否已经在缓存中，如果在缓存中，它将返回缓存的结果，而不需要重新计算。如果没有在缓存中，那么函数将被调用并且结果将被添加到缓存中。当缓存满了，最少使用的条目将被抛弃

```python
# AC时间: 1352ms
from functools import lru_cache

@lru_cache(maxsize = 128) # maxsize取决于题目中给的参数范围，不知道怎么设也可以maxsize=None
def cube(i):
    return i**3

def solv():
    N = int(input())
    ans = []
    for a in range(2,N+1):
        for b in range(2,a):
            for c in range(b,a):
                for d in range(c,a):
                    if cube(a) == cube(b) + cube(c) + cube(d):
                        ans.append((a,b,c,d))
    
    return ans

for a,b,c,d in solv():
    print(f"Cube = {a}, Triple = ({b},{c},{d})")
```

缓存的方法也可以用在递归中（例：斐波那契）：

```python
from functools import lru_cache 

@lru_cache(maxsize = 128) 
def f(n):
    if n <= 2:
        return 1
    else:
        return f(n-1)+f(n-2)
# 直接使用递归而不缓存的话复杂度是O(2^n)的恐怖级别，缓存牺牲了O(n)的空间复杂度以降低时间复杂度
```

dp的题目，或者涉及到需要处理二维表格的题目，尽量设置边界保护圈而非每次都判断是否越界，这样不仅程序更简洁，同时减少了习惯序列与计算机序列的转换（第一个数在计算机中序号为0，但加了保护圈之后序号就变成1了）

有些时候题目有多行输入，但是输入到一半其实你就出结果了，这时候你不用管剩下来的数据能不能输进去，直接输出结果结束程序也能AC

有些题涉及非常大的数据，题目要求你对结果取模后输出，但实际上你往往得在中间运算中就取模，否则数据会爆掉

```python
for i in range(3, 1000000+1):
    dp[i] = (2*dp[i-1] + dp[i-2])%32767
```

列表的remove方法具有时间复杂度O(n)，而且一边遍历列表一边弹出列表中元素也不是很好，所以一般还是另开一个列表来记录访问状态，或者直接利用题目所给数据的范围建一个保护圈

```python
for i in range(n):
  for j in range(m):
    if abs(a[i]-b[j])<=1:
      b[j] = 1000;
      cnt += 1
      break
```

利用lambda函数处理数据的多层排序：第一个参数小的放前面，第一个参数相同时，第二个参数放前面：

```python
hotels.sort(key=lambda x:(x[0],x[1]))
# 先按第一个参数排，再按第二个参数排
# 如果两个参数的排序方式不一样，可以使用多次排序，因为Python 的排序算法支持稳定排序，
# 这意味着当两个项目的第一级排序键相同时，它们的相对顺序不会改变
hotels.sort(key=lambda x: x[1], reverse=True)  # 按照x[1]降序排序
hotels.sort(key=lambda x: x[0])                # 按照x[0]升序排序
```

涉及日期的题目用datetime模块可以大大减少思考时间

这里我们给一道题目以及题解作为例子：

![{624A1941-640A-48B4-ACD2-0E06A492A68F}.png](624A1941-640A-48B4-ACD2-0E06A492A68F.png)

```python
import datetime

# Read input
day = input()
n = int(input())

# Parse the input date
date = datetime.datetime.strptime(day, '%Y-%m-%d').date()
# datetime.datetime.strptime()方法将步骤2中读取的日期字符串转换成日期对象。
# 但这个datetime对象包含了日期与时间信息，我们只需要日期信息，所以用.date()方法将其提取出来
# '%Y-%m-%d'是一个格式字符串，告诉Python解释器如何解析输入的日期字符串
# （其中%Y代表四位数的年份，%m代表月份，%d代表日期）

# Add n days
new_date = date + datetime.timedelta(days=n)
# 创建一个datetime.timedelta对象来表示n天的时间间隔，并将其加到原来的日期对象上，得到新的日期

# Print the resulting date in the format YYYY-MM-DD
print(new_date.strftime('%Y-%m-%d'))
# 使用strftime()方法将新的日期对象格式化回字符串形式，并按照YYYY-MM-DD的格式打印出来
```

python的集合运算（不知道之前有没有写过反正再强调一次）

并集

```python
union_set = set1 | set2
```

差集

```python
diff_set = set1 - set2 # set1中有而set2中没有的元素
```

python在处理浮点运算的时候可能会出现很神奇的情况

```python
a = float(1)
b = float(0)
print(-b/(2*a))
print((-b)/(2*a))
# 你本来肯定期望输出0.0，但是它实际上会输出-0.0，原因其实也很显然
# 神奇的是在真正进行运算的时候又不会出现这个问题
a = float(1.0)
b = float(1.0)
print((-a+b)/1.0)
print((a-b)/1.0)
# 这时候你发现他们都输出0.0
# 但是下面的代码输出-0.0
print(-(a-b)/1.0)
# 可以大致猜出是前面那个负号的缘故
# 所以基本上，如果分子里的东西涉及到运算了，你就不用管，只要括号括在最外面就没问题
# 但是如果分子真只有一个数的时候，把括号括在外面也没有用，这时候你只能：
if b == 0:
    b = -b
```

any(x)判断x对象是否为空对象，如果都为空、0、False，则返回False，如果不都为空、0、False，则返回True

all(x)如果all(x)参数x对象的所有元素不为0、''、False或者x为空对象，则返回True，否则返回False

有些地方如果能用数学公式做就尽量别用代码实现，说不定会超时。。。（典型的，比如求和）

不少函数在math内置模块里直接有，比如math.log2(n)，当然考试的时候还是有可能碰到你不记得的，可以用：

```python
import math
help(math)
```

在终端可以慢慢找（懒得找常用函数汇总了，实在不行只能自己写了）

判断回文：

```python
if s == s[::-1]
# 秒了
```
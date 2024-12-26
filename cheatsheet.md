# Cheatsheet

2024 fall, Complied by 章宇轩 物理学院

## 1. 算法与模板

### （1）Dijkstra算法

```python
import heapq
m,n,p = map(int,input().split())
roadmap = [input().split() for _ in range(m)]
graph = {}
steps = [(0,1),(0,-1),(1,0),(-1,0)]
for x in range(m):
    for y in range(n):
        if roadmap[x][y]!='#':
            subgraph = {}
            for dx,dy in steps:
                nx = x+dx
                ny = y+dy
                if 0<=nx<m and 0<=ny<n and roadmap[nx][ny]!='#':
                    subgraph[(nx,ny)]=abs(int(roadmap[nx][ny])-int(roadmap[x][y]))
            graph[(x,y)]=subgraph
def dijkstra(startx,starty,destix,destiy):
    if roadmap[startx][starty]=='#' or roadmap[destix][destiy]=='#':
        print('NO')
        return
    distances = {}
    for x in range(m):
        for y in range(n):
            distances[(x,y)]=float('inf')
    distances[(startx,starty)]=0
    priority_queue = [(0,startx,starty)]
    while priority_queue:
        distance,x,y = heapq.heappop(priority_queue)
        '''
        if distance>distances[(x,y)]:
            continue
        '''
        for neighbors,weight in graph[(x,y)].items():
            newdistance = distance+weight
            if newdistance<distances[neighbors]:
                distances[neighbors]=newdistance
                heapq.heappush(priority_queue,(newdistance,neighbors[0],neighbors[1]))
    if distances[(destix,destiy)]==float('inf'):
        print('NO')
    else:
        print(distances[(destix,destiy)])

for _ in range(p):
    startx,starty,destix,destiy = map(int,input().split())
    dijkstra(startx,starty,destix,destiy)
```

### （2）二分查找

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid  # 返回目标元素的索引
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # 如果未找到目标元素，返回 -1
```

### （3）区间问题

#### 区间合并：

给出一堆区间，要求**合并**所有**有交集的区间** （端点处相交也算有交集）。最后问合并之后的区间

【**步骤一**】：按照区间**左端点**从小到大排序。

【**步骤二**】：维护前面区间中最右边的端点为ed。从前往后枚举每一个区间，判断是否应该将当前区间视为新区间。

假设当前遍历到的区间为第i个区间 [li,ri]，有以下两种情况：

- li≤ed：说明当前区间与前面区间**有交集**。因此**不需要**增加区间个数，但需要设置 ed=max(ed,ri)。
- li>ed: 说明当前区间与前面**没有交集**。因此**需要**增加区间个数，并设置 ed=max(ed,ri)。

#### 选择不相交区间：

给出一堆区间，要求选择**尽量多**的区间，使得这些区间**互不相交**，求可选取的区间的**最大数量**。这里端点相同也算有重复。

【**步骤一**】：按照区间**右端点**从小到大排序。

【**步骤二**】：从前往后依次枚举每个区间。

假设当前遍历到的区间为第i个区间 [li,ri]，有以下两种情况：

- li≤ed：说明当前区间与前面区间有交集。因此直接跳过。
- li>ed: 说明当前区间与前面没有交集。因此选中当前区间，并设置 ed=ri。

#### 区间选点：

给出一堆区间，取**尽量少**的点，使得每个区间内**至少有一个点**（不同区间内含的点可以是同一个，位于区间端点上的点也算作区间内）。

转化为前一种问题

#### 区间覆盖问题：

给出一堆区间和一个目标区间，问最少选择多少区间可以**覆盖**掉题中给出的这段目标区间。

【**步骤一**】：按照区间左端点从小到大排序。

【**步骤二**】：**从前往后**依次枚举每个区间，在所有能覆盖当前目标区间起始位置start的区间之中，选择**右端点**最大的区间。

假设右端点最大的区间是第i个区间，右端点为ri。

最后将目标区间的start更新成ri。

#### 区间分组：

给出一堆区间，问最少可以将这些区间分成多少组使得每个组内的区间互不相交

【**步骤一**】：按照区间左端点从小到大排序。

【**步骤二**】：从**前往后**依次枚举每个区间，判断当前区间能否被放到某个现有组里面。

（即判断是否存在某个组的右端点在当前区间之中。如果可以，则不能放到这一组）

假设现在已经分了 m 组了，第 k 组最右边的一个点是 rk，当前区间的范围是 [Li,Ri] 。则：

如果$L_i \le r_k$ 则表示第 i 个区间无法放到第 k 组里面。反之，如果 Li>rk， 则表示可以放到第 k 组。

- 如果所有 m 个组里面没有组可以接收当前区间，则当前区间新开一个组，并把自己放进去。
- 如果存在可以接收当前区间的组 k，则将当前区间放进去，并更新当前组的 rk=Ri。

为了能快速的找到能够接收当前区间的组，我们可以使用**优先队列 （小顶堆）**。

优先队列里面记录每个组的右端点值，每次可以在 O(1) 的时间拿到右端点中的的最小值。

### （4）bfs

```python
from collections import deque
#deque的append,appendleft,pop,popleft方法
  
def bfs(start, end):    
    q = deque([(0, start)])  # (step, start)
    in_queue = {start}

    while q:
        step, front = q.popleft() # 取出队首元素
        if front == end:
            return step # 返回需要的结果，如：步长、路径等信息
        for node in nodes:
            if node not in in_queue:
                in_queue.add(node)
                q.append((step+1,node))
        # 将 front 的下一层结点中未曾入队的结点全部入队q，并加入集合in_queue设置为已入队
```

### （5）背包问题

#### 关于是否要求“恰好”的区别：

初始状态设为0，还是设为负无穷

#### 0-1背包

```python
N,B = map(int, input().split())
*p, = map(int, input().split())
*w, = map(int, input().split())

dp=[0]*(B+1)
for i in range(N):
    for j in range(B, w[i] - 1, -1):
        dp[j] = max(dp[j], dp[j-w[i]]+p[i])
            
print(dp[-1])
```

#### 完全背包

```python
n, a, b, c = map(int, input().split())
dp = [0]+[float('-inf')]*n

for i in range(1, n+1):
    for j in (a, b, c):
        if i >= j:
            dp[i] = max(dp[i-j] + 1, dp[i])

print(dp[n])
```

#### 多重背包（每个物品数量有上限）

最简单的思路是将多个同样的物品看成多个不同的物品，从而化为0-1背包。稍作优化：可以改善拆分方式，譬如将m个1拆成x_1,x_2,……,x_t个1，只需要这些x_i中取若干个的和能组合出1至m即可。最高效的拆分方式是尽可能拆成2的幂，也就是所谓“二进制优化”

```python
def binary_optimized_multi_knapsack(weights, values, quantities, capacity):
    n = len(weights)
    items = []
    # 将每个物品拆分成若干子物品
    for i in range(n):
        w, v, q = weights[i], values[i], quantities[i]
        k = 1
        while k < q:
            items.append((k * w, k * v))
            q -= k
            k <<= 1#位运算，相当于k*=2
        if q > 0:
            items.append((q * w, q * v))
    # 动态规划求解01背包问题
    dp = [0] * (capacity + 1)
    for w, v in items:
        for j in range(capacity, w - 1, -1):
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[capacity]
weights = [1, 2, 3]
values = [6, 10, 12]
quantities = [10, 5, 3]
capacity = 15
print(binary_optimized_multi_knapsack(weights, values, quantities, capacity)) # 输出: 120
```

### （6）子序列问题

如果只需要知道长度，直接用二分查找即可；如果还需要知道具体序列，那么需要结合dp

**最长不降子序列**

```python
#把下面的代码改成bisect_right
```

**最长单调递增子序列**

```python
import bisect
n = int(input())
*lis, = map(int, input().split())
dp = [1e9]*n
for i in lis:
    dp[bisect.bisect_left(dp, i)] = i
print(bisect.bisect_left(dp, 1e8))
```

```python
#维护一个数组，用于记录当前找到的最长子序列的末尾元素，并且在此基础上再维护一个额外的数组或列表，用来记录每个元素在最长单调子序列中的前驱信息。这样，当你完成处理之后，可以通过前驱信息追踪回去，得到具体的子序列。
from bisect import bisect_left


def longest_increasing_subsequence(nums):
    if not nums:
        return []

    # dp数组，用于存储当前找到的最长子序列的末尾元素
    dp = []
    # 存储每个元素的前驱索引
    prev_indices = [-1] * len(nums)
    # 存储每个位置的索引
    indices = []

    for i, num in enumerate(nums):
        # 使用二分查找找到插入位置
        pos = bisect_left(dp, num)

        # 如果pos等于dp的长度，说明num比dp中的所有元素都大
        if pos == len(dp):
            dp.append(num)
            if pos > 0:
                prev_indices[i] = indices[-1]  # 更新前驱索引
            indices.append(i)
        else:
            dp[pos] = num
            indices[pos] = i  # 更新当前pos位置的索引
            if pos > 0:
                prev_indices[i] = indices[pos - 1]  # 更新前驱索引

    # 重建最长递增子序列
    lis = []
    k = indices[-1]
    while k >= 0:
        lis.append(nums[k])
        k = prev_indices[k]

    # 反转序列，得到从小到大的顺序
    lis.reverse()

    return lis


# 示例
nums = [10, 9, 2, 5, 3, 7, 101, 18]
result = longest_increasing_subsequence(nums)
print("最长单调递增子序列:", result)
```

**最长不增子序列**

**最长单调递减子序列**：这两个把上面那两个取负就行

#### 变形：Dilworth定理

把序列分成不上升子序列的最少个数，等于序列的最长上升子序列长度。把序列分成不降子序列的最少个数，等于序列的最长下降子序列长度

#### 变形：如果是把序列分成严格递减子序列的个数呢？

稍微修改一下就行，因为dilworth定理的本质是正链和反链，反链是严格递减，那么正链就是不上升

### （7）质数

判断数量较少的数

```python
def is_prime(n):
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

判断数量很多的数

由于列表查找复杂度是O(n)，所以可以把结果保留在一个列表里然后用索引访问是否是质数，这样子会快一点

```python
#欧拉筛
def euler(r):
    prime = [0 for i in range(r+1)]
    common = []
    for i in range(2, r+1):
        if prime[i] == 0:
            common.append(i)
        for j in common:
            if i*j > r:
                break
            prime[i*j] = 1
            if i % j == 0:
                break
    return prime
#埃氏筛
def SieveOfEratosthenes(n, prime): 
    p = 2
    while (p * p <= n): 
        # If prime[p] is not changed, then it is a prime 
        if (prime[p] == True): 
            # Update all multiples of p 
            for i in range(p * 2, n + 1, p): 
                prime[i] = False
        p += 1

s = [True] * (10**2 + 1)
SieveOfEratosthenes(10**2, s)
print(s)
#建议自己先运行一遍看看0,1,2这种地方会不会出问题
```

### （8）循环队列

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

### （9）辅助栈

```python
stack1 = []
stack2 = []
def push(weight):
    stack1.append(weight)
    if not stack2:
        stack2.append(weight)
    elif weight<=stack2[-1]:
        stack2.append(weight)
def pop():
    if stack1:
        popout = stack1.pop(-1)
        if stack2[-1]==popout:
            popout2 = stack2.pop(-1)
def minweight():
    if stack2:
        print(stack2[-1])
while True:
    try:
        s = input()
    except EOFError:
        break
    s = s.split()
    if len(s)==1:
        if s[0]=='pop':
            pop()
        else:
            minweight()
    else:
        push(int(s[1]))
```

判断能否使用栈的标准：LIFO

### （10）单调栈

```python
def trap6(height):
    total_sum = 0
    stack = []
    current = 0
    while current < len(height):
        while stack and height[current] > height[stack[-1]]:
            h = height[stack[-1]]
            stack.pop()
            if not stack:
                break
            distance = current - stack[-1] - 1
            min_val = min(height[stack[-1]], height[current])
            total_sum += distance * (min_val - h)
        stack.append(current)
        current += 1
    return total_sum
```

### （11）生成下一个排列

```python
def next_permutation(nums):
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = len(nums) - 1
        while nums[j] <= nums[i]:
            j -= 1
        nums[i], nums[j] = nums[j], nums[i]
    nums[i + 1:] = reversed(nums[i + 1:])
    return nums
```

### （12）公共子序列

```python
#伪代码
if word_a[i]==word_b[j]:
    cell[i][j]=cell[i-1][j-1]+1
else:
    cell[i][j]=max(cell[i-1][j],cell[i][j-1])
```

### （13）公共子串

```python
#伪代码
if word_a[i]==word_b[j]:
    cell[i][j]=cell[i-1][j-1]+1
else:
    cell[i][j]=0
```

### （14）滑动窗口

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if not s:
            return 0
        MAX = 1
        n = len(s)
        index1 = 0
        index2 = 0
        dic = {s[0]:0}
        while True:
            index2+=1
            if index2==n:
                break
            if s[index2] not in dic or dic[s[index2]]<index1:
                dic[s[index2]]=index2
                MAX = max(MAX,index2-index1+1)
            else:
                index1 = dic[s[index2]]+1
                MAX = max(MAX,index2-index1+1)
                dic[s[index2]]=index2
        return MAX
```

### （15）堆（优先队列）

```python
import sys
import heapq
from collections import defaultdict
input = sys.stdin.readline
 
minH = []
maxH = []
 
ldict = defaultdict(int)
rdict = defaultdict(int)
 
n = int(input())
 
for _ in range(n):
    op, l, r = map(str, input().strip().split())
    l, r = int(l), int(r)
    if op == "+":
        ldict[l] += 1
        rdict[r] += 1
        heapq.heappush(maxH, -l)
        heapq.heappush(minH, r)
    else:
        ldict[l] -= 1
        rdict[r] -= 1
    
    '''
    使用 while 循环，将最大堆 maxH 和最小堆 minH 中出现次数为 0 的边界移除。
    通过比较堆顶元素的出现次数，如果出现次数为 0，则通过 heappop 方法将其从堆中移除。
    '''
    while len(maxH) > 0 >= ldict[-maxH[0]]:
        heapq.heappop(maxH)
    while len(minH) > 0 >= rdict[minH[0]]:
        heapq.heappop(minH)
    #这实际上是一种“懒删除”策略，事实上对于堆你也只能这么删（汗）
    
    
    '''
    判断堆 maxH 和 minH 是否非空，并且最小堆 minH 的堆顶元素是否小于
    最大堆 maxH 的堆顶元素的相反数。
    '''
    if len(maxH) > 0 and len(minH) > 0 and minH[0] < -maxH[0]:
        print("Yes")
    else:
        print("No")
```

### （16）懒更新

```python
for i in range(n):
    c = arr[i][1]
    cnt[c] += 1
    if vis[c]:
        while cnt[Q[0][1]]: # 懒更新，每次只更新到堆中的最小值是实际的最小值
            f = heapq.heappop(Q)
            f = (f[0] + cnt[f[1]], f[1])
            heapq.heappush(Q, f)
            cnt[f[1]] = 0
```

### （17）Greedy后悔

```python
import heapq


def max_potions(n, potions):
    # 当前健康值
    health = 0
    # 已经饮用的药水效果列表，用作最小堆
    consumed = []

    for potion in potions:
        # 尝试饮用当前药水
        health += potion
        heapq.heappush(consumed, potion)
        if health < 0:
            # 如果饮用后健康值为负，且堆中有元素
            if consumed:
                health -= consumed[0]
                heapq.heappop(consumed)


    return len(consumed)

n = int(input())
potions = list(map(int, input().split()))
print(max_potions(n, potions))
```

### （18）田忌赛马

```python
 lTian = 0; rTian = n - 1
    lKing = 0; rKing = n - 1
    ans = 0
    while lTian <= rTian:
        if Tian[lTian] > King[lKing]:
            ans += 1;
            lTian += 1
            lKing += 1
        elif Tian[rTian] > King[rKing]:
            ans += 1
            rTian -= 1
            rKing -= 1
        else:
            if Tian[lTian] < King[rKing]:
                ans -= 1
            
            lTian += 1
            rKing -= 1
```

### （19）Manacher算法：最长回文子串

```python
def manacher(s):
    s = '#' + '#'.join(s) + '#'
    n = len(s)
    p = [0] * n
    c = r = 0
    for i in range(n):
        mirr = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirr])
        while i + p[i] + 1 < n and i - p[i] - 1 >= 0 and s[i + p[i] + 1] == s[i - p[i] - 1]:
            p[i] += 1
        if i + p[i] > r:
            c, r = i, i + p[i]
    max_len, center_index = max((n, i) for i, n in enumerate(p))
    return s[center_index - max_len:center_index + max_len].replace('#', '')
```

### （20）Kadane算法：最长字串的和

```python
def kadane(l):
    dp = [0]*n
    dp[0]=l[0]
    for i in range(1,n):
        if dp[i-1]<0:
            dp[i]=l[i]
        else:
            dp[i]=dp[i-1]+l[i]
    return max(dp)
```

扩展：最大子矩阵

```python
for i in range(n):
    for j in range(i+1,n):
        copy = [sum(l[x][i:j+1]) for x in range(n)]
        MAX = kadane(copy)
        GLOMAX = max(GLOMAX,MAX)
```

### （21）Merge sort

```python
def MergeSort(arr):
    if len(arr)<=1: return arr
    else:
        l,r=arr[:len(arr)//2],arr[len(arr)//2:]
        return Merge(MergeSort(l),MergeSort(r))
def Merge(l,r):
    res=[]
    i,j=0,0
    while i<len(l) and j<len(r):
        if l[i]<=r[j]:
            res.append(l[i])
            i+=1
        else:
            res.append(r[j])
            j+=1
    res+=l[i:]+r[j:]
    return res
```

### （22）Quick sort

```python
def quickSort(arr):
    if len(arr)<=1:
        return arr
    else:
        mid=arr[len(arr)//2]
        l,m,r=[],[],[]
        for i in arr:
            if i < mid : l.append(i)
            elif i > mid : r.append(i)
            else : m.append(i)
        return quickSort(l) + m + quickSort(r)
```

### （23）关于dfs和bfs的一些理解

dfs和bfs都有一个很强大的特性，就是它的输出是直接满足“字典序”（感性理解一下）的

比如说，你要生成符合字典序的排列，你可以使用dfs

比如说，你要从一个数通过两个操作“H”和“O”最快到达另一个数，相同步数的操作比较字典序，这个时候你直接让H操作比O操作先入队即可，bfs程序的其他部分不需要做任何修改

注意，dfs和bfs的“顺序”是两种不同意义上的

bfs路径记录：

法一：直接加在queue的元组里面存着

法二：回溯思路

```python
def bfs(graph, start, goal):
    queue = deque([start])
    visited = set([start])
    path = {start: None}
    while queue:
        current = queue.popleft()
        if current == goal:
            return build_path(path, start, goal)
        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                path[neighbor] = current
    return None

def build_path(path, start, goal):
    current = goal
    path_list = [current]
    while current != start:
        current = path[current]
        path_list.append(current)
    path_list.reverse()
    return path_list
```

dfs路径记录：

```python
def dfs(graph, start, goal, path=None):
    if path is None:
        path = [start]
    if start == goal:
        return path
    for neighbor in graph[start]:
        if neighbor not in path:
            result = dfs(graph, neighbor, goal, path + [neighbor])
            if result:
                return result
    return None
```

### （24）dfs+缓存=动规

以滑雪为例，你既可以通过肉眼看出，可以使用sort来得到dp的顺序，也可以写一个dfs+缓存-缓存会自动帮你找到dp的顺序；Help Jimmy是滑雪的升级版

这种方法的重点是合理设计dfs函数的对应意义，使得缓存能在其上得到最有效的利用，也即，找到dp的”状态“

## 2.技巧

### （1）递归

```python
from functools import lru_cache
import sys
sys.setrecursionlimit(50000)
@lru_cache(maxsize=None)
def rec():
    ...
#如果有多个函数都需要缓存，需要将装饰器加在每个函数上
#不要加错位置！（这个很难debug）
```

### （2）enumrate

```python
nums = [5, 3, 9, 1, 6]
# 使用 enumerate 获取索引和值，然后根据值排序
sorted_pairs = sorted(enumerate(nums), key=lambda x: x[1])
#enumerate函数返回一个枚举对象，它将 nums 列表中的每个元素与其索引（即位置）配对。例如，对于列表 [5, 3, 9, 1, 6]，enumerate 会产生 (0, 5), (1, 3), (2, 9), (3, 1), (4, 6) 这样的元组。

# 解包排序后的索引和值
sorted_indices, sorted_nums = zip(*sorted_pairs)

# 将两个列表打包成一个新的元组列表也可以使用 zip 
l1 = [1, 2, 3]
l2 = [4, 5, 6]
packed = zip(l1, l2)
```

### （3）bisect

```python
import bisect
index1 = bisect.bisect_left(lst,number)
#index1左侧的所有元素都小于目标值，右侧的所有元素都大于等于目标值
index2 = bisect.bisect_right(lst,number)
#index2左侧的所有元素都小于等于目标值，右侧的所有元素都大于目标值
index = bisect.bisect(lst,number)
#二分查找
```

### （4）global

在函数内部，当想要改变全局变量的值时使用

OJ的pylint是静态检查，有时候报的不对，第一行加# pylint: skip-file可解决

### （5）一些有用的函数、技巧

```python
eval()#将字符串视为python表达式进行运算
print(f'{numbers} card(s)')#格式化输出
print(f'{number:.5f}')#保留5位小数
import math
math.ceil()#向正无穷取整
math.floor()#向负无穷取整
int()#去尾
s.lower()#将所有大写字母转化为小写字母，返回新字符串；类似的还有upper()
ord()#返回ascii
chr()#返回字符
s.find(sub)#查找一个子字符串在另一个字符串中的位置。如果找到了子字符串，则会返回子字符串的第一个字符在主字符串中的索引位置；如果没有找到子字符串，返回-1（只会返回第一个搜到的）
s.count(sub)#统计某个字串出现的次数

#如果在循环中用break语句提前终止循环，循环变量将停留在它终止时候的值；当循环结束后，循环变量依然可以在循环外部访问，并且保持它最后被赋的值

from math import gcd
print(gcd(a,b))#最大公约数内置函数
s.strip()#移除首尾的特定字符，默认是空白符，比如想移除'-'就s.strip('-')
#类似的还有lstrip移除开头的，rstrip移除结尾的

#在使用split()方法时，如果字符串中没有对应的分隔符，将返回一个只包含原字符串单一元素的列表；如果分隔符出现在字符串的首尾或者连续出现，那么 split() 方法将会把这些分隔符视为分隔点，并产生相应的空字符串项

s_reversed=s[::-1]#字符串反转
#其它进制转十进制
decimal = int(string,digits) #string为其他进制字符串，digits为进制，比如2、4、8
#十进制转其他进制
binary = bin(num)
octal = oct(num)
hexadecimal = hex(num)
# 但要注意这样输出的是有前缀的，比如'0b1010'，所以要先切片

MIN=float('inf')
MAX=float('-inf')# 生成无限大无限小

#处理字符串时可以控制字符添加在原有字符串的左侧还是右侧
s1 = s+'test'
s2 = 'test'+s

#python的int自动满足大整数，甚至可以把’000123’这种字符串转成’123’

#输入字符串转成列表
l = list(input())
```

### （6）try...except

```python
while True:
    try:
        input()
    except EOFError:
        break
    ...
```

### （7）set

```python
#并集
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set3 = set1.union(set2)
set1 = {1, 2, 3}
set2 = {3, 4, 5}
set1.update(set2)
#差集（set1中有而set2中没有的元素）
diff_set = set1 - set2 # set1中有而set2中没有的元素
#删除特定元素
set1.remove()
set1.discard()
#处理元素不存在的情况，remove()会报错，而discard()什么都不干
```

### （8）列表

```python
newlist.append(l.pop(index))#列表的pop方法可以返回pop的东西
list1+[' ']#列表可以直接拼接

numbers.sort()#sort方法无返回值
sorted_numbers = sorted(numbers)#sorted函数有返回值
#reverse=True降序
#默认以第一个元素为标准进行排序

#python内置sort具有稳定性，如果想要对第一个元素排序，然后第一个元素相同情况下对第二个元素降序排序
#这时用reverse=True是不行的，因为reverse的底层逻辑是正常排一遍然后反过来
lst = [(1, 5), (2, 3), (1, 2), (2, 1), (1, 3)]
sorted_lst = sorted(lst, key=lambda x: (x[0], -x[1]))
#如果第二个元素为字符串不能取负，则需要用额外的方法
from functools import cmp_to_key
# 自定义比较函数
def compare_items(x, y):
    if x[0] < y[0]:
        return -1
    elif x[0] > y[0]:
        return 1
    elif x[1] < y[1]:  # 降序排序
        return 1
    elif x[1] > y[1]:
        return -1
    else:
        return 0

lst = [(1, 'z'), (2, 'c'), (1, 'a'), (2, 'b'), (1, 'y')]
sorted_lst = sorted(lst, key=cmp_to_key(compare_items))
#或者还可以分布进行，把需要倒序的放前面
lst.sort(key=lambda x: x[1], reverse=True)  # 按照x[1]降序排序
lst.sort(key=lambda x: x[0])             

l.remove(value)#移除列表中第一个匹配指定值的元素。如果列表中不存在该值，则会抛出一个 ValueError 异常
```

### （9）字典

```python
for key in my_dict:
for key in my_dict.keys():#遍历字典的键
for key in my_dict,values():#遍历字典的值
for key,value in my_dict.items():#遍历键值对
#以上的操作都是对一个视图对象进行遍历，并非是副本，因此它是可以实时改变字典内部内容的
if key in my_dict:#检测是否存在于字典的键中
#时间复杂度O(1)

#对字典进行排序
from collections import OrderedDict

my_dict = {'b': 2, 'c': 3, 'a': 1}
ordered_dict = OrderedDict(sorted(my_dict.items()))
print(ordered_dict)  # 输出: OrderedDict([('a', 1), ('b', 2), ('c', 3)])

#字典可以删除键值对O(1)
my_dict = {'a': 1, 'b': 2, 'c': 3}
del my_dict['b']
```

### （10）输入输出节约时间

```python
import sys
readin = sys.stdin.read().strip()
l = readin.split()
#一口气读进所有输入然后慢慢处理
#把答案都存在列表ans中
print('\n'.join(ans))
```

### （11）列表推导式

```python
#创建二维列表
l = [[] for _ in range(n)]#l = [[]]*n用这种方法还会碰到副本的问题

#加判断的列表推导式
lnew = [str(x) for x in lold if x > n]
```

### （12）lambda函数

```python
l3 = list(map(lambda x, y: x + y, l1, l2))
# lambda x, y: x + y 是一个匿名函数，接受两个参数x和y，并返回它们的和
# map()函数会对l1和l2中的每个元素应用这个匿名函数
```

将一个比较函数转换为一个key函数

```python
from functools import cmp_to_key
def comparison(x):
    if x[-1]=='M':
        return float(x[:-1])*1
    else:
        return float(x[:-1])*1000
key_func = cmp_to_key(comparison)
sorted_list = sorted(my_list, key=key_func)
```

### （13）for…else，while…else

在循环正常结束时执行else内的代码块，这里的“正常结束”指的是没有通过 break 语句提前终止循环

### （14）Counter

把一个列表中的重复数据组合起来，以{’key’:counted_times}的字典形式储存，时间复杂度O(n)，可以在后期减少很多遍历的耗时

```python
#统计众数的出现次数
from collections import Counter
print(max(Counter(input().split()).values()))
```

### （15）help

不少函数在math内置模块里有，如果考试的时候碰到记不得的可以用

```python
import math
help(math)
```

在终端慢慢找

### （16）product（笛卡尔积）

```python
from itertools import product
products = product(['0','1'],repeat=6)
for product in products:
    ...
```

### （17）permutation（排列）

```python
from itertools import permutations
permutation = permutations('ABCD',4)
for per in permutation:
    ...
```

### （18）前缀和

p[0]=a[0],p[i]=a[0]+a[1]+...+a[i]

## 3.注意点

图相关问题的行与列不要搞混

慎用字符串拼接做字典的键

在dfs中，特别特别要当心list的传递，养成传递副本的习惯，否则怎么也查不出错误

涉及到排序的题，注意检查reverse=True写了没，以及到底题目数据有没有排过序

注意数字的字符串形式和数值形式

用print调试完程序之后print不要忘记注释掉！！！

在很多时候python的运算精度其实是不够的，比如求三次根号n**(1/3)，你代入n=1000，它返回9.9999999998；所以能用乘法别用根号之类的运算

remove时间复杂度为O(n)，能不remove就留着，牺牲空间复杂度

有些题涉及非常大的数据，题目要求你对结果取模后输出，但实际上你往往得在中间运算中就取模，否则数据会爆掉

二维数组需要深度拷贝：import copy；A=copy.deepcopy(X)

有时候也可以考虑考虑brute force?比如一些保证有解的题目；或者说很难找到给出最值的方案，但是很容易判断一个值可不可以取到（这个时候可以用二分）（brute force与计算机思维）

python中比较数字和比较字符串的逻辑是不一样的，比如，11>9，'11'<'9'（逐位比较），看你需要哪一种

为了省事相似的代码块可以直接复制，但是但是！！！这种时候很容易变量改不干净！即使是简单题也容易WA

有些题目使用了lru_cache，视情况函数需要定义在循环体内部：比如Help Jimmy这道题，每一次循环都是一个全新的图，因此你必须重置你的lru_cache

## 4.每日选做里一些题的思路

#### 1.炸鸡排、电池寿命

思路：最理想情况是每块鸡排都炸完,用时为所有鸡排用时的平均值 如果耗时最长的鸡排比其他鸡排的平均用时更长,说明无论其他鸡排怎么摆都会使那块鸡排炸不完, 那索性那块鸡排就一直放那炸,考虑剩下的鸡排 否则就可以达到最理想情况:平均用时

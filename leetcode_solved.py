수프밍 문제 풀이
Reverse Linked List 링크드 리스트가 재밌는듯
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        node, prev = head, None
        while node:
            # next를 prev로 연결하면서 반복하면 됨(그림으로 그리는 게 젤 빠르게 이해됨)
            next, node.next = node.next, prev
            prev, node = node, next
        #이때 prev를 리턴해야함
        return prev

Remove Duplicates from Sorted Array
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        # 리스트로 풀이
        if len(nums)==0: return 0
        i=0 
        for j in range(1,len(nums)):
            if nums[i]!=nums[j]: # 다르면
                i+=1  # i를 바꿔주고
                nums[i]=nums[j] # Swap
        
        return i+1
Remove Element
class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i = 0 # index로
        while nums.count(val):
            nums.remove(val)
        return len(nums)
        # 되게 쉬움
Next permutation
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        flag = True # boolean으로 시작
        for i in range(len(nums)-1, 0, -1):
            if nums[i] > nums[i-1]:
                index = i-1
                flag = False
                break
        if flag: nums.sort() #sort()하는 곳
        else:
            swapNum = float('inf')
            swapIndex = -1
            for i in range(index+1, len(nums)):
                if nums[i] > nums[index] and swapNum > nums[index]:
                    swapNum = nums[i]
                    swapIndex = i
            temp = nums[swapIndex]
            nums[swapIndex] = nums[index]
            nums[index] = temp 
            tail = sorted(nums[index+1:])
            nums[:] = nums[:index+1] + tail
Search in Rotated Sorted Array
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        # l<=r 일 때 해야됨
        while left <= right:
            mid = left + ((right - left) // 2)
            
            if target == nums[mid]:
                return mid
            
            if nums[left] <= nums[mid]:
                if target > nums[mid] or target < nums[left]:
                    left = mid + 1
                else:
                    right = mid - 1
            
            else:
                if target < nums[mid] or target > nums[right]:
                    right = mid - 1
                else:
                    left = mid + 1
            
        return -1
Find First and Last Position of Element in Sorted Array
class Solution(object):
    def searchRange(self, nums, target):
        if not nums or target not in nums:
            return [-1, -1]
        # first와 last를 적절히 찾아서 대입
        def first_occ(nums, target):
            left, right = -1, len(nums)
            while left + 1 < right:
                mid = (left + right) // 2
                if nums[mid] < target:
                    left = mid
                else:
                    right = mid

            return right

        def last_occ(nums, target):
            left, right = -1, len(nums)
            while left + 1 < right:
                mid = (left + right) // 2
                if nums[mid] <= target:
                    left = mid
                else:
                    right = mid

            return left

        first = first_occ(nums, target)
        last = last_occ(nums, target)

        return [first, last]
Search Insert Position
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left=0
        right=len(nums)-1
        while left<=right:
            mid=(left+right)//2
            if nums[mid]>target: # left 찾고
                right=mid-1
            elif nums[mid]<target: # right
                left=mid+1   
            else: # 올바른 값을 찾을 때
                return mid
        #틀리면
        return left 
Longest Common Prefix
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs: # empty면
            return ''
        # enumerate를 쓰는 풀이 남겨놓음
        minS = min(strs, key = len)
        for i, x in enumerate(minS):
            for oth in strs:
                if oth[i] !=x:
                    return minS[:i]
        return minS
Valid Parentheses
class Solution:
    def isValid(self, s: str) -> bool:
        # 스택을 사용할 것임
        stack=[]
        brackets={'}':'{',')':'(',']':'['}
        for bracket in s:
            if bracket in brackets.values(): 
                stack.append(bracket)
            else:
                if stack and brackets[bracket]==stack[-1] :  
                    stack.pop()
                else: 
                    return False
        
        if stack:
            return False
        return True
Find the Index of the First Occurrence in a String
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        # base 처리 
        if not needle:
            return 0
        
        h_len, n_len = len(haystack), len(needle)
        # +1 중요
        for i in range(h_len -  n_len + 1):
            if haystack[i:i+n_len] == needle:
                return i
        return -1
Length of Last Word 
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        # split() 써서
        word = s.strip().split()
        if not word:
            return 0
        
        return len(word[-1])
Add Binary
class Solution:
    def addBinary(self, a: str, b: str) -> str:
	# format으로 적어둠
        return format(int(a,2)+int(b,2), 'b')
Valid Palindrome
class Solution:
    def isPalindrome(self, s: str) -> bool:
        # 일단 낮추고
        s = s.lower()
        # regEx로 알파벳 제외 없애고
        s = re.sub('[^a-z0-9]', '', s)
        return s == s[::-1] 
Excel Sheet Column Title
class Solution(object):
    def convertToTitle(self, columnNumber):
        res = ""
        while columnNumber > 0:
            output = chr(ord('A') + (columnNumber - 1) % 26) + res
            # -1 : A(65) 관련
            columnNumber = (columnNumber - 1) // 26
        return res
Excel Sheet Column Number
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        ans = 0
        N = len(columnTitle)
        for n in range(N-1, -1, -1):
            ans += 26**n * (ord(columnTitle[N-n-1])-ord('A')+1)
            
        return ans
Merge Two Sorted Lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if (not list1) or (list2 and list1.val > list2.val):
            list1, list2 = list2, list1
        if list1:
            list1.next = self.mergeTwoLists(list1.next, list2)
        return list1
Plus One
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        tmp = ""
        for num in digits: tmp += str(num)
        tmp = str(int(tmp) + 1)
        ans = []
        for i in tmp: ans.append(int(i))
        return ans
 Sqrt(x)
import math
# math로 하면 바로 메서드 있음
class Solution:
    def mySqrt(self, x: int) -> int:
        ans = int(math.sqrt(x))
        return ans
Climbing Stairs
class Solution:
    def climbStairs(self, n: int) -> int:
        # 기본 케이스
        if n <= 2:
            return n
        #피보나치 느낌?
        ans = [0] * n
        ans[0] = 1
        ans[1] = 2
        for i in range(2, n):
            ans[i] = ans[i - 1] + ans[i - 2]
        return ans[-1]
 Merge Sorted Array
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
         del nums1[m:] 
         del nums2[n:] 
         nums1 += nums2 
         nums1.sort() 
        """
        Do not return anything, modify nums1 in-place instead.
        """
Binary Tree Inorder Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        ans =[]

        def travel(cur):
            if cur.left is not None: travel(cur.left) 
            ans.append(cur.val) 
            if cur.right is not None: travel(cur.right) 
        
        if root is not None:
            travel(root)
        
        return ans
Same Tree
# 찾아낸 풀이 -> DFS라고 함(공부용으로 저장)
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.val != q.val:
            return False
        
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)

Symmetric Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    # recursive하게 체커 함수로 확인
    def checker(self, p, q):
        if not p and not q: return True
        if not p or not q: return False
        if p.val != q.val: return False
        return self.checker(p.left, q.right) and self.checker(p.right, q.left)
    def isSymmetric(self, root: TreeNode) -> bool:
        return self.checker(root.left, root.right)
Maximum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # 끝까지 내려가본다는 마인드로 구현 -> 재귀형태로
        if root == None: return 0
        if root.left == None and root.right == None: return 1
        elif root.left == None: return self.maxDepth(root.right) + 1
        elif root.right == None: return self.maxDepth(root.left) + 1
        else: return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
Convert Sorted Array to Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> TreeNode:
        if not nums:
            return None
        mid = len(nums) // 2
        node = TreeNode(nums[mid])
        node.left = self.sortedArrayToBST(nums[:mid])
        node.right = self.sortedArrayToBST(nums[mid + 1 :])

        return node
Balanced Binary Tree
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        # Depth First Search를 바탕으로 구현(balance 확인할 때 용이하다고 함)
        def dfs(root):
            if not root: return [True, 0]

            left, right = dfs(root.left), dfs(root.right)
            balance = (left[0] and right[0] and 
            abs(left[1] -right[1]) <=1)

            return [balance, 1+max(left[1], right[1])]

        return dfs(root)[0] 
Minimum Depth of Binary Tree
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDepth(self, root: Optional[TreeNode]) -> int:
        # 간단하게 재귀로 
        if not root:
            return 0
        if not root.right:
            return 1+self.minDepth(root.left)
        if not root.left:
            return 1+self.minDepth(root.right)
        return 1+min(self.minDepth(root.left), self.minDepth(root.right))
Path Sum
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:

        def dfs(root, x):
            if not root:
                return False

            x +=  root.val
            if not root.left and not root.right:
                return x == targetSum

            return dfs(root.left, x) or dfs(root.right, x)
        
        return dfs(root, 0)
Pascal's Triangle
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        ans = [[1]] # 좋은 풀이라 남겨놓았음
        for n in range(1, numRows):
            nextRow = [1]
            
            for m in range(n-1):
                nextRow.append(ans[n-1][m]+ans[n-1][m+1])
            
            nextRow.append(1)
            ans.append(nextRow)
            
        return ans
Linked List Cycle
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        s, f = head, head
        
        while f and f.next:
            s = s.next
            f = f.next.next
            if s == f:
                return True
            
        return False
Reverse Bits
class Solution:
    def reverseBits(self, n: int) -> int:
        ans = bin(n)[2:]
        while len(ans) < 32:
            ans = '0'+ans # '0'을 더하기
        return int('0b' + ans[::-1], 2)

 Number of 1 Bits 완전 쉬움
class Solution:
    def hammingWeight(self, n: int) -> int:
        return str(bin(n)).count('1')
Happy Number
class Solution:
    def isHappy(self, n: int) -> bool:
        done = set()
        while True:
            ans = 0
            for x in str(n):
                done.add(int(x))
                ans += int(x)*int(x)
            if ans == 1:
                return True
            if ans in done:
                return False
            n = ans

Remove Linked List Elements

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        ans = ListNode(-1)
        ans.next = head
        cur_node = ans
        while cur_node.next!= None:
            if cur_node.next.val == val:
                cur_node.next = cur_node.next.next
            else:
                cur_node = cur_node.next  
        return ans.next   
Isomorphic Strings 재밌음
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        # 사실 set 이용하면 매우 간편하지 않나?
        return len(set(zip(s,t))) == len(set(s)) == len(set(t))

Contains Duplicate
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        return len(nums) != len(set(nums))

Count Complete Tree Nodes
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
def counter(root: Optional[TreeNode]):
    if root == None: return 0
    # 1더하고 해야함
    return 1+ counter(root.left) + counter(root.right)

class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        left = 0
        right = 0
        rile = riri = root
        while riri != None:
            rile = rile.left
            left += 1
        while riri != None:
            riri = riri.right
            right += 1
        return counter(root) if (left!=right) else (2**(left)-1)
Implement Stack using Queues
import queue

class MyStack:
    def __init__(self):
        self.q = queue.Queue()

    def push(self, x: int) -> None:
        self.q.put(x)

    def pop(self) -> int:
        for _ in range(self.q.qsize() - 1):
            self.q.put(self.q.get())
        return self.q.get()

    def top(self) -> int:
        for _ in range(self.q.qsize() - 1):
            self.q.put(self.q.get())
        peek = self.q.get()
        self.q.put(peek)
        return peek

    def empty(self) -> bool:
        return self.q.qsize() == 0
Invert Binary Tree
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                node.left, node.right = node.right, node.left
                stack.append(node.left)
                stack.append(node.right)
        return root
Power of Two
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and n & (n-1) == 0
Implement Queue using Stacks

class MyQueue:

    def __init__(self):
        self.stack_in = list()
        self.stack_out = list()

    def push(self, x: int) -> None:
        self.stack_in.append(x)

    def pop(self) -> int:
        length = len(self.stack_in) - 1
        for _ in range(length):
            self.stack_out.append(self.stack_in.pop())
        front_value = self.stack_in.pop()
        for _ in range(length):
            self.stack_in.append(self.stack_out.pop())   
        return front_value

    def peek(self) -> int:
        return self.stack_in[0]

    def empty(self) -> bool:
        if self.stack_in:
            return False
        return True
# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
Palindrome Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        tmp = []
        cur_node = head

        while cur_node:
            tmp.append(cur_node.val)
            cur_node = cur_node.next

        if tmp == list(reversed(a)):
            return True

        return False
Valid Anagram
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if sorted(s) != sorted(t):
            return False
        
        return True
Binary Tree Paths
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def binaryTreePaths(self, root: TreeNode) -> List[str]:
        def dfs(node, path):
            path += '->'
            path += str(node.val)
            if not node.left and not node.right:
                return ans.append(path[2:])
            if node.left:
                dfs(node.left, path)
            if node.right:
                dfs(node.right, path)
        ans = []
        dfs(root, "")
        return ans

Add Digits
class Solution:
    def addDigits(self, num: int) -> int:
        num = list(map(int, list(str(num))))
        while true :
            num = sum(num)
            if num in range(0, 10) :
                return num
            else :
                num = list(map(int, list(str(num))))
Ugly Number
class Solution:
    def isUgly(self, n: int) -> bool:
        if n < 1: return False
        while n % 2 == 0:
            n = n / 2
        while n % 3 == 0:
            n = n / 3
        while n % 5 == 0:
            n = n / 5
        if n > 1:
            return False
        return True
Missing Number
못풂
First Bad Version
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:
class Solution:
    def firstBadVersion(self, n: int) -> int:
        start = 1
        end = n
        while start <= end:
            middle = (start + end) // 2
            if isBadVersion(middle):
                end = middle - 1
            else:
                start = middle + 1
        return start
        
Move Zeroes
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        count=nums.count(0)
        nums[:]=[i for i in nums if i != 0]
        nums+=[0]*count
        
Word Pattern
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        word = s.split(' ')
        w_dict = {}
        if len(pattern) != len(word):
            return False
        for i, d in enumerate(pattern):
            if d in w_dict and w_dict[d] != word[i]:
                return False
            elif d not in w_dict and word[i] in w_dict.values():
                return False
            elif d not in w_dict:
                w_dict[d] = word[i]
        return True 
Range Sum Query - Immutable
class NumArray:

    def __init__(self, nums: List[int]):
        self.sumList = [0]*(len(nums)+1)
        for i in range(len(nums)):
            self.sumList[i+1] = self.sumList[i] + nums[i]

    def sumRange(self, left: int, right: int) -> int:
        return self.sumList[right+1] - self.sumList[left]
Power of Three
class Solution:
    def isPowerOfThree(self, n: int) -> bool:
        return n > 0 and pow(3, 31, n) == 0
countBits
class Solution:
    def countBits(self, n: int) -> List[int]:
        dp = [0]

        for i in range(1, n+1):
            dp.append(dp[i&(i-1)]+1)
        return dp
Power of Four
class Solution:
    def isPowerOfFour(self, n: int) -> bool:
    return True if n > 0 and n & (n - 1) == 0 and n % 3 == 1 else False

Reverse String
class Solution:
    def reverseString(self, s: List[str]) -> None:
        l = 0
        r = len(s) - 1
        while l < r:
            s[l], s[r] = s[r], s[l]
            l += 1
            r -= 1
Reverse Vowels of a String
class Solution:
    def reverseVowels(self, s: str) -> str:
        string_list = list(s)
        i = 0
        j = len(string_list) - 1
        vowel = ["a", "e", "i", "o", "u", "A", "E", "I", "O", "U"]

        while True:
            while i <= j and string_list[i] not in vowel:
                i += 1
            while i <= j and string_list[j] not in vowel:
                j -= 1

            if i >= j:
                break

            string_list[i], string_list[j] = string_list[j], string_list[i]
            i += 1
            j -= 1
        return "".join(string_list)
Intersection of Two Arrays II
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        a = set(nums1)
        b = set(nums2)
        s = a & b # 교집합
        ans = []
        for num in s:
            count = min(nums1.count(num), nums2.count(num))
            ans.extend([num] * count) # 개수를 세서 그만큼 배열에 추가

        return ans
Valid Perfect Square
class Solution:
    def isPerfectSquare(self, num: int) -> bool:
        left, right = 1, num
        
        while left <= right:
            mid = left + (right - left) // 2
            mid_square = mid * mid
            
            if mid_square == num:
                return True
            elif mid_square < num:
                left = mid + 1
            else:
                right = mid - 1
                
        return False
Guess Number Higher or Lower
class Solution: def guessNumber(self, n: int) -> int: left, right = 1, n while left <= right: mid = left + (right - left) // 2 result = guess(mid) if result == 0: return mid elif result == -1: right = mid - 1 else: left = mid + 1 return -1
Ransom Note
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        from collections import Counter
        
        ransom_counter = Counter(ransomNote)
        magazine_counter = Counter(magazine)
        
        for char, count in ransom_counter.items():
            if magazine_counter[char] < count:
                return False
        
        return True
First Unique Character in a String
class Solution:
    def firstUniqChar(self, s: str) -> int:
        from collections import Counter
        
        count = Counter(s)
        for idx, char in enumerate(s):
            if count[char] == 1:
                return idx
        return -1
Find the Difference
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        from collections import Counter
        
        count_s = Counter(s)
        count_t = Counter(t)
        
        for char in count_t:
            if count_t[char] != count_s.get(char, 0):
                return char
Is Subsequence
class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        if not s:
            return True
        
        s_index, t_index = 0, 0
        
        while t_index < len(t):
            if t[t_index] == s[s_index]:
                s_index += 1
                if s_index == len(s):
                    return True
            t_index += 1
        
        return s_index == len(s)
 Binary Watch
class Solution:
    def readBinaryWatch(self, turnedOn: int) -> List[str]:
        results = []
        for h in range(12):
            for m in range(60):
                if bin(h).count('1') + bin(m).count('1') == turnedOn:
                    results.append(f"{h}:{m:02d}")
        return results
Sum of Left Leaves
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        def isLeaf(node):
            return node is not None and node.left is None and node.right is None
        
        def dfs(node):
            if node is None:
                return 0
            sum_left_leaves = 0
            if node.left and isLeaf(node.left):
                sum_left_leaves += node.left.val
            sum_left_leaves += dfs(node.left)
            sum_left_leaves += dfs(node.right)
            return sum_left_leaves
        
        return dfs(root)
Convert a Number to Hexadecimal
class Solution:
    def toHex(self, num: int) -> str:
        if num == 0:
            return "0"
        
        hex_chars = "0123456789abcdef"
        result = []
        
        # Handle negative numbers using 2's complement
        if num < 0:
            num += 2 ** 32
        
        while num > 0:
            result.append(hex_chars[num % 16])
            num //= 16
        
        return ''.join(result[::-1])

Longest Palindrome
class Solution:
    def longestPalindrome(self, s: str) -> int:
        from collections import Counter
        
        count = Counter(s)
        length = 0
        odd_found = False
        
        for freq in count.values():
            if freq % 2 == 0:
                length += freq
            else:
                length += freq - 1
                odd_found = True
        
        if odd_found:
            length += 1
        
        return length
Fizz Buzz
class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        result = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                result.append("Fizz")
            elif i % 5 == 0:
                result.append("Buzz")
            else:
                result.append(str(i))
        return result
Third Maximum Number
class Solution:
    def thirdMax(self, nums: List[int]) -> int:
        nums = set(nums)
        if len(nums) < 3:
            return max(nums)
        nums.remove(max(nums))
        nums.remove(max(nums))
        return max(nums)
 Add Strings
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        i, j = len(num1) - 1, len(num2) - 1
        carry = 0
        result = []
        
        while i >= 0 or j >= 0 or carry:
            n1 = int(num1[i]) if i >= 0 else 0
            n2 = int(num2[j]) if j >= 0 else 0
            
            total = n1 + n2 + carry
            carry = total // 10
            result.append(str(total % 10))
            
            i -= 1
            j -= 1
        
        return ''.join(result[::-1])
Number of Segments in a String
class Solution:
    def countSegments(self, s: str) -> int:
        return len(s.split())
Arranging Coins
class Solution:
    def arrangeCoins(self, n: int) -> int:
        left, right = 0, n
        
        while left <= right:
            mid = left + (right - left) // 2
            if mid * (mid + 1) // 2 == n:
                return mid
            if mid * (mid + 1) // 2 < n:
                left = mid + 1
            else:
                right = mid - 1
        
        return right
Find All Numbers Disappeared in an Array
class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        for num in nums:
            index = abs(num) - 1
            if nums[index] > 0:
                nums[index] = -nums[index]
        
        return [i + 1 for i in range(len(nums)) if nums[i] > 0]
Assign Cookies
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        g.sort()
        s.sort()
        
        child_i = 0
        cookie_i = 0
        
        while child_i < len(g) and cookie_i < len(s):
            if s[cookie_i] >= g[child_i]:
                child_i += 1
            cookie_i += 1
        
        return child_i
Repeated Substring Pattern
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return (s + s)[1:-1].find(s) != -1
Hamming Distance
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')
 Island Perimeter
class Solution:
    def islandPerimeter(self, grid: List[List[int]]) -> int:
        rows, cols = len(grid), len(grid[0])
        perimeter = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 1:
                    perimeter += 4
                    if r > 0 and grid[r-1][c] == 1:
                        perimeter -= 2
                    if c > 0 and grid[r][c-1] == 1:
                        perimeter -= 2
                        
        return perimeter

Number Complement
class Solution:
    def findComplement(self, num: int) -> int:
        bit_length = num.bit_length()
        mask = (1 << bit_length) - 1
        return num ^ mask
 License Key Formatting
class Solution:
    def licenseKeyFormatting(self, s: str, k: int) -> str:
        s = s.replace('-', '').upper()
        size = len(s)
        first_group = size % k
        parts = []

        if first_group:
            parts.append(s[:first_group])
        
        for i in range(first_group, size, k):
            parts.append(s[i:i+k])
        
        return '-'.join(parts)
Max Consecutive Ones
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        max_count = 0
        current_count = 0
        
        for num in nums:
            if num == 1:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
Construct the Rectangle
class Solution:
    def constructRectangle(self, area: int) -> List[int]:
        w = int(area**0.5)
        while area % w != 0:
            w -= 1
        l = area // w
        return [l, w]
Teemo Attacking
class Solution:
    def findPoisonedDuration(self, timeSeries: List[int], duration: int) -> int:
        if not timeSeries:
            return 0
        
        total_duration = 0
        
        for i in range(1, len(timeSeries)):
            total_duration += min(timeSeries[i] - timeSeries[i - 1], duration)
        
        return total_duration + duration
Next Greater Element I
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        next_greater = {}
        stack = []
        
        for num in nums2:
            while stack and stack[-1] < num:
                next_greater[stack.pop()] = num
            stack.append(num)
        
        return [next_greater.get(num, -1) for num in nums1]
Keyboard Row
class Solution:
    def findWords(self, words: List[str]) -> List[str]:
        row1 = set("qwertyuiopQWERTYUIOP")
        row2 = set("asdfghjklASDFGHJKL")
        row3 = set("zxcvbnmZXCVBNM")
        
        result = []
        
        for word in words:
            if all(char in row1 for char in word) or all(char in row2 for char in word) or all(char in row3 for char in word):
                result.append(word)
        
        return result
Find Mode in Binary Search Tree
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        from collections import Counter
        
        def inorder(node):
            if node is None:
                return []
            return inorder(node.left) + [node.val] + inorder(node.right)
        
        if not root:
            return []
        
        count = Counter(inorder(root))
        max_freq = max(count.values())
        
        return [k for k, v in count.items() if v == max_freq]

Base 7
class Solution:
    def convertToBase7(self, num: int) -> str:
        if num == 0:
            return "0"
        
        negative = num < 0
        num = abs(num)
        result = []
        
        while num:
            result.append(str(num % 7))
            num //= 7
        
        if negative:
            result.append('-')
        
        return ''.join(result[::-1])
Relative Ranks
class Solution:
    def findRelativeRanks(self, score: List[int]) -> List[str]:
        sorted_scores = sorted(score, reverse=True)
        rank_dict = {score: str(i + 1) for i, score in enumerate(sorted_scores)}
        
        for i in range(len(sorted_scores)):
            if i == 0:
                rank_dict[sorted_scores[i]] = "Gold Medal"
            elif i == 1:
                rank_dict[sorted_scores[i]] = "Silver Medal"
            elif i == 2:
                rank_dict[sorted_scores[i]] = "Bronze Medal"
        
        return [rank_dict[s] for s in score]
 Perfect Number
class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        if num <= 1:
            return False
        
        divisors_sum = 1
        sqrt_num = int(num ** 0.5)
        
        for i in range(2, sqrt_num + 1):
            if num % i == 0:
                divisors_sum += i
                if i != num // i:
                    divisors_sum += num // i
        
        return divisors_sum == num
Fibonacci Number
class Solution:
    def fib(self, n: int) -> int:
        if n <= 1:
            return n
        
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        
        return b
Detect Capital
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        return word.isupper() or word.islower() or word.istitle()

Longest Uncommon Subsequence I
class Solution:
    def findLUSlength(self, a: str, b: str) -> int:
        if a == b:
            return -1
        else:
            return max(len(a), len(b))
Minimum Absolute Difference in BST
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        def inorder(node):
            return inorder(node.left) + [node.val] + inorder(node.right) if node else []
        
        values = inorder(root)
        return min(abs(a - b) for a, b in zip(values, values[1:]))
Reverse String II
class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        result = []
        for i in range(0, len(s), 2 * k):
            part1 = s[i:i + k][::-1]
            part2 = s[i + k:i + 2 * k]
            result.append(part1 + part2)
        return ''.join(result)
Diameter of Binary Tree
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.diameter = 0
        
        def depth(node):
            if not node:
                return 0
            left = depth(node.left)
            right = depth(node.right)
            self.diameter = max(self.diameter, left + right)
            return max(left, right) + 1
        
        depth(root)
        return self.diameter
Student Attendance Record I
class Solution:
    def checkRecord(self, s: str) -> bool:
        return s.count('A') <= 1 and 'LLL' not in s
Reverse Words in a String III
class Solution:
    def reverseWords(self, s: str) -> str:
        return ' '.join(word[::-1] for word in s.split())
Maximum Depth of N-ary Tree
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

class Solution:
    def maxDepth(self, root: 'Node') -> int:
        if not root:
            return 0
        if not root.children:
            return 1
        return 1 + max(self.maxDepth(child) for child in root.children)
Array Partition
class Solution:
    def arrayPairSum(self, nums: List[int]) -> int:
        nums.sort()
        return sum(nums[i] for i in range(0, len(nums), 2))
 Binary Tree Tilt
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findTilt(self, root: TreeNode) -> int:
        self.total_tilt = 0
        
        def sum_and_tilt(node):
            if not node:
                return 0
            left_sum = sum_and_tilt(node.left)
            right_sum = sum_and_tilt(node.right)
            self.total_tilt += abs(left_sum - right_sum)
            return node.val + left_sum + right_sum
        
        sum_and_tilt(root)
        return self.total_tilt
Reshape the Matrix
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        flat_list = [num for row in mat for num in row]
        if len(flat_list) != r * c:
            return mat
        
        return [flat_list[i * c:(i + 1) * c] for i in range(r)]
Subtree of Another Tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        if not root:
            return False
        if self.isSameTree(root, subRoot):
            return True
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
    
    def isSameTree(self, s: TreeNode, t: TreeNode) -> bool:
        if not s and not t:
            return True
        if not s or not t:
            return False
        if s.val != t.val:
            return False
        return self.isSameTree(s.left, t.left) and self.isSameTree(s.right, t.right)
Distribute Candies
class Solution:
    def distributeCandies(self, candyType: List[int]) -> int:
        return min(len(set(candyType)), len(candyType) // 2)
N-ary Tree Postorder Traversal
# Definition for a Node.
class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children if children is not None else []

class Solution:
    def postorder(self, root: 'Node') -> List[int]:
        result = []
        
        def traverse(node):
            if not node:
                return
            for child in node.children:
                traverse(child)
            result.append(node.val)
        
        traverse(root)
        return result
Longest Harmonious Subsequence
class Solution:
    def findLHS(self, nums: List[int]) -> int:
        from collections import Counter
        count = Counter(nums)
        longest = 0
        
        for num in count:
            if num + 1 in count:
                longest = max(longest, count[num] + count[num + 1])
        
        return longest
Can Place Flowers
class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        count = 0
        length = len(flowerbed)
        
        for i in range(length):
            if flowerbed[i] == 0 and (i == 0 or flowerbed[i - 1] == 0) and (i == length - 1 or flowerbed[i + 1] == 0):
                flowerbed[i] = 1
                count += 1
                if count >= n:
                    return True
        
        return count >= n
Merge Two Binary Trees
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def mergeTrees(self, t1: TreeNode, t2: TreeNode) -> TreeNode:
        if not t1:
            return t2
        if not t2:
            return t1
        
        merged = TreeNode(t1.val + t2.val)
        merged.left = self.mergeTrees(t1.left, t2.left)
        merged.right = self.mergeTrees(t1.right, t2.right)
        return merged
Average of Levels in Binary Tree
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def averageOfLevels(self, root: TreeNode) -> List[float]:
        from collections import deque
        result = []
        queue = deque([root])
        
        while queue:
            level_sum = 0
            level_count = len(queue)
            for _ in range(level_count):
                node = queue.popleft()
                level_sum += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level_sum / level_count)
        
        return result
Maximum Average Subarray I
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        current_sum = sum(nums[:k])
        max_sum = current_sum
        
        for i in range(k, len(nums)):
            current_sum += nums[i] - nums[i - k]
            max_sum = max(max_sum, current_sum)
        
        return max_sum / k
Set Mismatch
class Solution:
    def findErrorNums(self, nums: List[int]) -> List[int]:
        num_set = set(nums)
        n = len(nums)
        duplicate = sum(nums) - sum(num_set)
        missing = sum(range(1, n + 1)) - sum(num_set)
        return [duplicate, missing]
Two Sum IV - Input is a BST
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findTarget(self, root: TreeNode, k: int) -> bool:
        def inorder(node):
            return inorder(node.left) + [node.val] + inorder(node.right) if node else []
        
        nums = inorder(root)
        left, right = 0, len(nums) - 1
        
        while left < right:
            total = nums[left] + nums[right]
            if total == k:
                return True
            elif total < k:
                left += 1
            else:
                right -= 1
        
        return False
Robot Return to Origin
class Solution:
    def judgeCircle(self, moves: str) -> bool:
        x, y = 0, 0
        for move in moves:
            if move == 'U':
                y += 1
            elif move == 'D':
                y -= 1
            elif move == 'L':
                x -= 1
            elif move == 'R':
                x += 1
        return x == 0 and y == 0
Image Smoother
class Solution:
    def imageSmoother(self, img: List[List[int]]) -> List[List[int]]:
        rows, cols = len(img), len(img[0])
        result = [[0] * cols for _ in range(rows)]
        
        for r in range(rows):
            for c in range(cols):
                count = 0
                sum_val = 0
                for i in range(r - 1, r + 2):
                    for j in range(c - 1, c + 2):
                        if 0 <= i < rows and 0 <= j < cols:
                            sum_val += img[i][j]
                            count += 1
                result[r][c] = sum_val // count
        
        return result
Second Minimum Node In a Binary Tree
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findSecondMinimumValue(self, root: TreeNode) -> int:
        self.first_min = root.val
        self.second_min = float('inf')
        
        def dfs(node):
            if node:
                if self.first_min < node.val < self.second_min:
                    self.second_min = node.val
                elif node.val == self.first_min:
                    dfs(node.left)
                    dfs(node.right)
        
        dfs(root)
        return self.second_min if self.second_min < float('inf') else -1
Longest Continuous Increasing Subsequence
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        max_length = 1
        current_length = 1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 1
        
        return max_length
Valid Palindrome II
class Solution:
    def validPalindrome(self, s: str) -> bool:
        def is_palindrome_range(i, j):
            return all(s[k] == s[j - k + i] for k in range(i, j))
        
        for i in range(len(s) // 2):
            if s[i] != s[~i]:
                j = len(s) - 1 - i
                return is_palindrome_range(i + 1, j) or is_palindrome_range(i, j - 1)
        
        return True
 Baseball Game
class Solution:
    def calPoints(self, ops: List[str]) -> int:
        stack = []
        
        for op in ops:
            if op == '+':
                stack.append(stack[-1] + stack[-2])
            elif op == 'D':
                stack.append(2 * stack[-1])
            elif op == 'C':
                stack.pop()
            else:
                stack.append(int(op))
        
        return sum(stack)
Binary Number with Alternating Bits
class Solution:
    def hasAlternatingBits(self, n: int) -> bool:
        bits = bin(n)[2:]
        return all(bits[i] != bits[i + 1] for i in range(len(bits) - 1))
Count Binary Substrings
class Solution:
    def countBinarySubstrings(self, s: str) -> int:
        groups = [1]
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                groups[-1] += 1
            else:
                groups.append(1)
        
        count = 0
        for i in range(1, len(groups)):
            count += min(groups[i - 1], groups[i])
        
        return count
Degree of an Array
class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        left, right, count = {}, {}, defaultdict(int)
        
        for i, num in enumerate(nums):
            if num not in left:
                left[num] = i
            right[num] = i
            count[num] += 1
        
        degree = max(count.values())
        min_length = len(nums)
        
        for num in count:
            if count[num] == degree:
                min_length = min(min_length, right[num] - left[num] + 1)
        
        return min_length
Search in a Binary Search Tree
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def searchBST(self, root: TreeNode, val: int) -> TreeNode:
        if root is None or root.val == val:
            return root
        if val < root.val:
            return self.searchBST(root.left, val)
        return self.searchBST(root.right, val)
Kth Largest Element in a Stream
import heapq

class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.min_heap = nums
        heapq.heapify(self.min_heap)
        while len(self.min_heap) > k:
            heapq.heappop(self.min_heap)

    def add(self, val: int) -> int:
        if len(self.min_heap) < self.k:
            heapq.heappush(self.min_heap, val)
        elif val > self.min_heap[0]:
            heapq.heapreplace(self.min_heap, val)
        return self.min_heap[0]
Binary Search
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1
Design HashSet
class MyHashSet:

    def __init__(self):
        self.bucket_size = 1000
        self.buckets = [[] for _ in range(self.bucket_size)]

    def add(self, key: int) -> None:
        bucket_index = key % self.bucket_size
        if key not in self.buckets[bucket_index]:
            self.buckets[bucket_index].append(key)

    def remove(self, key: int) -> None:
        bucket_index = key % self.bucket_size
        if key in self.buckets[bucket_index]:
            self.buckets[bucket_index].remove(key)

    def contains(self, key: int) -> bool:
        bucket_index = key % self.bucket_size
        return key in self.buckets[bucket_index]

# Your MyHashSet object will be instantiated and called as such:
# obj = MyHashSet()
# obj.add(key)
# obj.remove(key)
# param_3 = obj.contains(key)
Design HashMap
class MyHashMap:

    def __init__(self):
        self.bucket_size = 1000
        self.buckets = [[] for _ in range(self.bucket_size)]

    def put(self, key: int, value: int) -> None:
        bucket_index = key % self.bucket_size
        for i, (k, v) in enumerate(self.buckets[bucket_index]):
            if k == key:
                self.buckets[bucket_index][i] = (key, value)
                return
        self.buckets[bucket_index].append((key, value))

    def get(self, key: int) -> int:
        bucket_index = key % self.bucket_size
        for k, v in self.buckets[bucket_index]:
            if k == key:
                return v
        return -1

    def remove(self, key: int) -> None:
        bucket_index = key % self.bucket_size
        for i, (k, v) in enumerate(self.buckets[bucket_index]):
            if k == key:
                self.buckets[bucket_index].pop(i)
                return

# Your MyHashMap object will be instantiated and called as such:
# obj = MyHashMap()
# obj.put(key,value)
# param_2 = obj.get(key)
# obj.remove(key)
To Lower Case
class Solution:
    def toLowerCase(self, s: str) -> str:
        return s.lower()
1-bit and 2-bit Characters
class Solution:
    def isOneBitCharacter(self, bits: List[int]) -> bool:
        i = 0
        while i < len(bits) - 1:
            if bits[i] == 1:
                i += 2
            else:
                i += 1
        return i == len(bits) - 1
 Find Pivot Index
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        total_sum = sum(nums)
        left_sum = 0
        
        for i, num in enumerate(nums):
            if left_sum == total_sum - left_sum - num:
                return i
            left_sum += num
        
        return -1
Self Dividing Numbers
class Solution:
    def selfDividingNumbers(self, left: int, right: int) -> List[int]:
        def is_self_dividing(num):
            for digit in str(num):
                if digit == '0' or num % int(digit) != 0:
                    return False
            return True
        
        result = []
        for num in range(left, right + 1):
            if is_self_dividing(num):
                result.append(num)
        
        return result
Flood Fill
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        oldColor = image[sr][sc]
        if oldColor == newColor:
            return image
        
        def dfs(r, c):
            if r < 0 or r >= len(image) or c < 0 or c >= len(image[0]) or image[r][c] != oldColor:
                return
            image[r][c] = newColor
            dfs(r + 1, c)
            dfs(r - 1, c)
            dfs(r, c + 1)
            dfs(r, c - 1)
        
        dfs(sr, sc)
        return image
Find Smallest Letter Greater Than Target
class Solution:
    def nextGreatestLetter(self, letters: List[str], target: str) -> str:
        start, end = 0, len(letters)-1

        if target>= letters[-1] or target < letters[0]:
            return letters[0]

        while start <= end:
            mid = (start+end)//2
            if letters[mid]<=target:
                start = mid+1
            else:
                end = mid-1
        return letters[start]
Min Cost Climbing Stairs
class Solution:
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        cost = cost
        dp = [0 for _ in range(len(cost))]
        dp[0] = cost[0]
        dp[1] = cost[1]
        for j in range(2, len(cost)):
            dp[j] = min(dp[j-2] + cost[j], dp[j-1] + cost[j])
        return min(dp[len(dp)-1], dp[len(dp)-2])
Largest Number At Least Twice of Others
class Solution:
    def dominantIndex(self, nums: List[int]) -> int:
        max_num = max(nums)
        check_num = max_num // 2
        
        check = [num for num in nums if check_num < num]
        
        if len(set(check)) > 1:
            return -1
        else:
            return nums.index(max_num)
Prime Number of Set Bits in Binary Representation
class Solution:
    def countPrimeSetBits(self, L: int, R: int) -> int:
        return sum(bin(i).count('1') in [2,3,5,7,11,13,17,19] for i in range(L, R+1))
Toeplitz Matrix
class Solution:
    def isToeplitzMatrix(self, matrix: List[List[int]]) -> bool:
        for i in range(len(matrix)-1):
            if matrix[i][:-1] != matrix[i+1][1:]:
                return False
        return True
 Jewels and Stones
class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        stone_dict = defaultdict(int)
        count = 0
        for stone in stones:
            stone_dict[stone] += 1

        for jewel in jewels:
            count += stone_dict[jewel]

        return count
Minimum Distance Between BST Nodes
class Solution:
    answer = sys.maxsize
    prev = -sys.maxsize
    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        if root.left:
            self.minDiffInBST(root.left)
        self.answer = min(self.answer, root.val - self.prev)
        self.prev = root.val
        if root.right:
            self.minDiffInBST(root.right)
        return self.answer
Rotate String
class Solution:
    def rotateString(self, s: str, goal: str) -> bool:
        answer = False
        s_list = list(s)

        shift_word_list = []

        while True:
            if goal == f"{''.join(shift_word_list)[::-1]}{''.join(s_list)}":
                answer = True
                break

            if not s_list:
                break

            shift_word_list.append(s_list.pop())

        return answer
 Unique Morse Code Words
class Solution:
    @staticmethod
    def transformation(text):
        morse_dict = {'a': '.-', 'b': '-...', 'c': '-.-.', 'd': '-..', 'e': '.', 'f': '..-.', 'g': '--.', 'h': '....', 'i': '..', 'j': '.---', 'k': '-.-', 'l': '.-..', 'm': '--', 'n': '-.', 'o': '---', 'p': '.--.', 'q': '--.-', 'r': '.-.', 's': '...', 't': '-', 'u': '..-', 'v': '...-', 'w': '.--', 'x': '-..-', 'y': '-.--', 'z': '--..'}
        
        char_list = list(text)
        
        temp = ""
        
        for char in char_list:
            temp = temp + morse_dict[char]
            
        return temp
    
    
    def uniqueMorseRepresentations(self, words: List[str]) -> int:
        morse_list = []
        
        for word in words:
            morse = self.transformation(word)
            
            morse_list.append(morse)
            
        set_morse = set(morse_list)
        
        return len(set_morse)
 Most Common Word
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        real_para = ""
        
        for letter in paragraph:
            if not letter.isalpha():
                letter = ' '
            else:
                letter = letter.lower()
            real_para += letter
        
        paragraph = real_para.split()
        paragraph = collections.Counter(paragraph)
        
        for word in paragraph.most_common():
            if word[0] not in banned:
                return word[0]
Shortest Distance to a Character
class Solution:
    def shortestToChar(self, S: str, C: str) -> List[int]:
        s_list = list(S)
        
        idx_list = []
        
        for i in range(len(s_list)):
            if s_list[i] == C:
                idx_list.append(i)
        
        answer_list = []
        
        for i in range(len(s_list)):
            temp = min([ abs(i-idx) for idx in idx_list ])
            
            answer_list.append(temp)
            
        return answer_list
Positions of Large Groups
class Solution:
    def largeGroupPositions(self, s: str) -> List[List[int]]:
        answer = [] 
        i = 0 # 그룹의 시작 
        for j in range(len(s)): 
            if j == len(s) -1  or s[j] != s[j+1]:
                if j-i+1 >= 3:
                    answer.append([i,j])
                i = j+1
        return answer
 Flipping an Image
try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class Solution(object):
    def flipAndInvertImage(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        for row in A:
            for i in xrange((len(row)+1) // 2):
                row[i], row[~i] = row[~i] ^ 1, row[i] ^ 1
        return A
Backspace String Compare
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        s_que, t_que = deque(s), deque(t)
        s_stack, t_stack = [], []

        while s_que:
            s1 = s_que.popleft()
            if s1!='#':
                s_stack.append(s1)

            else:
                if len(s_stack)!=0:
                    s_stack.pop()

        while t_que:
            t1 = t_que.popleft()
            if t1!='#':
                t_stack.append(t1)
            else:
                if len(t_stack)!=0:
                    t_stack.pop()

        return True if s_stack==t_stack else False
Buddy Strings
class Solution:
    def buddyStrings(self, A: str, B: str) -> bool:
        if len(A) != len(B) : return False
        elif A == B : 
            setA = list(set(list(A)))
            if len(setA) < len(A) : return True
            else : return False
        
        diff = []
        for i in range(len(A)) :
            if A[i] != B[i] : 
                if len(diff) == 2 : return False
                else : diff.append(i)
        
        if len(diff) < 2 : return False
        listA = list(A)
        temp = listA[diff[0]]
        listA[diff[0]] = listA[diff[1]]
        listA[diff[1]] = temp
        return ''.join(listA) == B
Lemonade Change
class Solution(object):
    def lemonadeChange(self, bills):
        five, ten = 0, 0

        for dollar in bills:
            if dollar == 5:
                five += 1
            elif dollar == 10:
                if five == 0:
                    return False
                five -= 1
                ten += 1
            else:
                if five > 0 and ten > 0:
                    five -= 1
                    ten -= 1
                elif five >= 3:
                    five -= 3
                else:
                    return False
        return True
Transpose Matrix
class Solution:
    def transpose(self, matrix: List[List[int]]) -> List[List[int]]:
        H, W = len(matrix), len(matrix[0])
        newM = [[0 for _ in range(H)] for _ in range(W)]
        
        for h in range(H):
            for w in range(W):
                newM[w][h] = matrix[h][w]
                
        return newM
Binary Gap
class Solution:
  def binaryGap(self, n: int) -> int:
    ans = 0
    d = -32  # the distance between any two 1s

    while n:
      if n & 1:
        ans = max(ans, d)
        d = 0
      n //= 2
      d += 1

    return ans
Leaf-Similar Trees
class Solution:
    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        leaf1, leaf2 = [], []

        def get_leaves(root, leaf):
            if not root:
                return 

            if root.left:
                get_leaves(root.left, leaf)
            if root.right:
                get_leaves(root.right, leaf)
            if not root.left and not root.right:
                leaf.append(root.val)

        get_leaves(root1, leaf1)
        get_leaves(root2, leaf2)

        return True if leaf1 == leaf2 else False
Middle of the Linked List
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        N = 0
        nhead = head
        while nhead.next:
            nhead = nhead.next
            N += 1
        
        half = N//2 + N%2
        while half > 0:
            head = head.next
            half -= 1
        
        return head
Uncommon Words from Two Sentences
from collections import Counter

class Solution:
    def uncommonFromSentences(self, A: str, B: str) -> List[str]:
        array = A.split(" ") + B.split(" ")
        
        cnt = list(Counter(array).items())
        
        answer = [item[0] for item in cnt if item[1] == 1]
        
        return answer
Monotonic Array
class Solution:
    def isMonotonic(self, A: List[int]) -> bool:
        reverse_flag = False
        
        if max(A) == A[0]:
            reverse_flag = True
        
        sorted_A = sorted(A, reverse=reverse_flag)
        
        if sorted_A == A:
            return True
        
        return False
Sort Array By Parity
class Solution:
    def sortArrayByParity(self, nums: List[int]) -> List[int]:
        return sorted(nums, key=lambda x: x%2)
Smallest Range I
class Solution:
    def smallestRangeI(self, A: List[int], K: int) -> int:
        return max(max(A) - min(A) - 2 * K, 0)
Reverse Only Letters
class Solution:
    def reverseOnlyLetters(self, s: str) -> str:
        res = ""
        def findalpha(reverseidx) : 
            while reverseidx > 0 and not s[reverseidx].isalpha() : 
                reverseidx-=1
            return reverseidx
        reverseidx = findalpha(len(s)-1)
        for char in s : 
            if char.isalpha() : 
                res += s[reverseidx]
                reverseidx = findalpha(reverseidx-1)
            else : 
                res += char
        return res
Long Pressed Name
class Solution:
    def isLongPressedName(self, name, typed):
        nameSplit = self.chrSpliter(name)
        typedSplit = self.chrSpliter(typed)

        if (len(nameSplit) != len(typedSplit)):
            return False

        for i in range(len(nameSplit)):
            if nameSplit[i][0] != typedSplit[i][0] or len(nameSplit[i]) > len(typedSplit[i]):
                return False
        return True
        
     def chrSpliter(self, chr):
        chrList = []
        temp = ""
        for i in range(len(chr)):
            temp += chr[i]

            if i == len(chr) - 1 or chr[i] != chr[i + 1]:
                chrList.append(temp)
                temp = ""
        return chrList
Unique Email Addresses
import re
class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        unique_emails= []
        for i in range(len(emails)):
            email_front, email_end = emails[i].split('@')
            print(email_front, email_end)
            
            email_front = email_front.replace(".", "")
            print(email_front, email_end)
            
            
            email_front_without_localname = email_front.split('+')[0]
            
            email_add = email_front_without_localname + '@' + email_end
            
            if email_add not in unique_emails:
                unique_emails.append(email_add)
            print(unique_emails)
        return len(unique_emails)
Range Sum of BST
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    range_sum = 0

    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if root:
            if low <= root.val <= high:
                self.range_sum += root.val

            if low <= root.val:
                self.rangeSumBST(root.left, low, high)
            if high >= root.val:
                self.rangeSumBST(root.right, low, high)

        return self.range_sum
Valid Mountain Array
class Solution:
    def validMountainArray(self, arr: List[int]) -> bool:
        temp = False 
        valid = False 
        n = len(arr) 
        i = 0 
        # 올라가는 경우 
        # 인덱스가 범위 내이고 앞의 숫자가 다음 숫자보다 작음 
        while i + 1 < n and arr[i] < arr[i+1]: 
            i += 1 
            valid = True # 올라가는 케이스가 있었는지를 검증 
            
        # 내려가는 경우 
        # 인덱스가 범위 내이고 앞의 숫자가 다음 숫자보다 크고 올라가는 케이스가 존재했음
        while i + 1 < n and arr[i] > arr[i+1] and valid == True: 
            i += 1 
            temp = True 
        
        # 인덱스 끝까지 돌았는지 확인 
        if i+1 != n :
            temp = False
            
        return temp
DI String Match
class Solution:
    def diStringMatch(self, s: str) -> List[int]:
        start, end = 0, len(s)
        answer = []
        
        for i in range(len(s)):
            if s[i] == "I":
                answer.append(start)
                start += 1
            
            else:
                answer.append(end)
                end -= 1
        
        answer.append(start)
        # answer.append(end) # start = end
        
        return answer
Verifying an Alien Dictionary
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        characterOrder = dict()
        i = 0
        for ch in order:
            characterOrder[ch] = i
            i += 1
        def compare(a, b):
            for i in range(len(a) if len(a) < len(b) else len(b)):
                if a[i] != b[i]:
                    return characterOrder[a[i]] - characterOrder[b[i]]
            return len(a) - len(b)
        std = sorted(words, key=cmp_to_key(compare))
        
        if len(std) != len(words): return False
        for i in range(len(std)):
            if std[i] != words[i]: return False
        return True
 Univalued Binary Tree
class Solution(object):
    def isUnivalTree(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """

        # TreeNode 가 존재하지 않을 때 True
        if not root:
            return True

        # root의 left 가 존재하고 root의 val 과 root의 left의 val 값이 다르면
        # return False
        if root.left and root.val != root.left.val:
            return False

        # root의 right 가 존재하고 root의 val과 root의 right의 val 값이 다르면
        # return False
        if root.right and root.val != root.right.val:
            return False

        # root의 left와 right의 val이 root의 val 과 같다면
        # left right를 완전 탐색하여 True 이면 True
        # False 이면 return False
        return self.isUnivalTree(root.left) and self.isUnivalTree(root.right)
Squares of a Sorted Array
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        result = [0] * len(nums)
        left, right = 0, len(nums)-1

        for i in range(len(nums)-1, -1, -1):
            if abs(nums[left]) > abs(nums[right]):
                result[i] = nums[left] **2
                left +=1
            else:
                result[i] = nums[right] **2
                right -=1

        return result
Find Common Characters
from collections import Counter

class Solution:
    def commonChars(self, A: List[str]) -> List[str]:
        compare = collections.Counter(A[0])  
        
        for i in range(len(A)):
            cnt = collections.Counter(A[i])
            compare = compare & cnt
            
        answer = list(compare.elements())
                
        return answer
Maximize Sum Of Array After K Negations
class Solution:
    def largestSumAfterKNegations(self, A: List[int], K: int) -> int:
        for i in range(K):
            if K == 0:
                break
            min_val = min(A)
            index = A.index(min_val)
            if min_val < 0:
                K -= 1
                A[index] = -min_val
            elif (K % 2) == 0:
                break
            else:
                K -= 1
                A[index] = -min_val
        return sum(A)
 Complement of Base 10 Integer
class Solution:
    def bitwiseComplement(self, n: int) -> int:
        if n == 0:  return 1
        ans = 0
        cur = 1
        
        while n > 1:
            a, b = n//2, n%2
            n = a
            if b == 0:
                ans += cur
            cur *= 2
        
        return ans
Remove Outermost Parentheses
class Solution:
    def removeOuterParentheses(self, s: str) -> str:
        stack = []
        count = 0
        outer = False
        flag = True
        
        for ch in s:
            if ch == '(':
                stack.append(ch)
                count += 1
                if count >= 1 and flag:
                    stack.pop()
                    outer = True
                    flag = False
            else:
                stack.append(ch)
                count -= 1
                if outer and count == 0:
                    stack.pop()
                    outer = False
                    flag = True
                    
        return "".join(map(str, stack))
Divisor Game
class Solution:
    def divisorGame(self, n: int) -> bool:
        cnt=0
        while n!=1:
            n-=1
            cnt+=1
        
        if cnt%2==0:
            return False
        else:
            return True
Last Stone Weight
import heapq

class Solution:
    def lastStoneWeight(self, stones: list[int]) -> int:
        stones = [-s for s in stones]
        heapq.heapify(stones)

        while len(stones) >1:
            first = heapq.heappop(stones)
            second = heapq.heappop(stones)
            if second > first:
                heapq.heappush(stones, first-second)

        stones.append(0) 
        return abs(stones[0])  
Remove All Adjacent Duplicates In String
class Solution:
    def removeDuplicates(self, s: str) -> str:
        res = ""
        st  = deque()
        for i in range(len(s)):
            if st :
                if st[-1] == s[i] : 
                    st.pop()
                else : 
                    st.append(s[i])
            else : 
                st.append(s[i])
        while st : 
            res+=st.popleft()
        return res
Height Checker
class Solution:
    def heightChecker(self, heights: List[int]) -> int:
        new_h = sorted(heights)
        match = 0
        for i in range(len(heights)):
            if heights[i] != new_h[i]:
                match += 1
                
        return match
Number of Steps to Reduce a Number to Zero
class Solution:
    def numberOfSteps(self, num: int) -> int:
        steps = 0

        while num != 0:
            steps += 1
            if num & 1:
                num -= 1
            else:
                num >>= 1

        return steps

The K Weakest Rows in a Matrix
class Solution(object):
    def kWeakestRows(self, mat, k):
        """
        :type mat: List[List[int]]
        :type k: int
        :rtype: List[int]
        """

        for i in range(len(mat)):
            mat[i] = [i, sum(mat[i])]

        mat.sort(key=lambda x: x[1])

        return [mat[i][0] for i in range(k)]
Check If N and Its Double Exist
class Solution:
    def checkIfExist(self, arr: List[int]) -> bool:
        for i in range(len(arr)):
            for j in range(len(arr)):
                if i != j:
                    if arr[i] == 2 * arr[j]:
                        return True
        return False
Count Negative Numbers in a Sorted Matrix
class Solution:
    def countNegatives(self, grid: List[List[int]]) -> int:
        ans = 0
        m = len(grid)
        n = len(grid[0])
        
        for i in range(m):
            if i < 0:
                ans += (m - i) * n
                break
            for j in range(n):
                if grid[i][j] < 0:
                    ans += (n - j)
                    break
        
        return ans
Sort Integers by The Number of 1 Bits
class Solution:
    def sortByBits(self, arr: List[int]) -> List[int]:
        def countOneBit(n):
            stack = []
            while n > 1:
                stack.append(n % 2)
                n //= 2
            stack.append(n)
            return stack.count(1)
        
        return list(sorted(arr, key = lambda x : (countOneBit(x), x)))
Number of Days Between Two Dates
import datetime

class Solution:
    def daysBetweenDates(self, date1: str, date2: str) -> int:
        date_01 = datetime.datetime.strptime(date1, '%Y-%m-%d')
        date_02 = datetime.datetime.strptime(date2, '%Y-%m-%d')
        
        answer = abs(date_01- date_02).days
        
        return answer
How Many Numbers Are Smaller Than the Current Number
class Solution:
    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        answer = []
        
        for num in nums:
            temp =  [n for n in nums if n < num]
            
            answer.append(len(temp))
            
        return answer
Generate a String With Characters That Have Odd Counts
class Solution(object):
    def generateTheString(self, n):
        """
        :type n: int
        :rtype: str
        """
        result = ['a']*(n-1)
        result.append('a' if n%2 else 'b')
        return "".join(result)
Find a Corresponding Node of a Binary Tree in a Clone of That Tree
class Solution:
    def getTargetCopy(self, original: TreeNode, cloned: TreeNode, target: TreeNode) -> TreeNode:
        c_start=cloned
        stack=[c_start]
        while stack:
            node=stack.pop()
            if node.val==target.val:
                return node
            if node.left:stack.append(node.left)
            if node.right:stack.append(node.right)
Lucky Numbers in a Matrix
class Solution:
    def luckyNumbers (self, matrix: List[List[int]]) -> List[int]:
        matrix_len = len(matrix[0])
        row_len = len(matrix)
        for i in range(matrix_len):
            temp = []
            for j in range(row_len):
                temp.append(matrix[j][i])
                
            for k in range(row_len):
                if max(temp) == min(matrix[k]):
                    return [max(temp)]
Find the Distance Value Between Two Arrays
from typing import List

class Solution:
    def findTheDistanceValue(self, arr1: List[int], arr2: List[int], d: int) -> int:
        # solution one: 暴力
        res = 0
        for x in arr1:
            cnt = 0
            for y in arr2:
                if abs(x-y) <= d:
                    break
                else:
                    cnt += 1
                if cnt == len(arr2):
                    res += 1
        return res

        # solution two: 一行代码
        return sum(all(abs(a1 - a2) > d for a2 in arr2) for a1 in arr1)

if __name__ == "__main__":
    arr1 = [-803,715,-224,909,121,-296,872,807,715,407,94,-8,572,90,-520,-867,485,-918,-827,-728,-653,-659,865,102,-564,-452,554,-320,229,36,722,-478,-247,-307,-304,-767,-404,-519,776,933,236,596,954,464]
    arr2 = [817,1,-723,187,128,577,-787,-344,-920,-168,-851,-222,773,614,-699,696,-744,-302,-766,259,203,601,896,-226,-844,168,126,-542,159,-833,950,-454,-253,824,-395,155,94,894,-766,-63,836,-433,-780,611,-907,695,-395,-975,256,373,-971,-813,-154,-765,691,812,617,-919,-616,-510,608,201,-138,-669,-764,-77,-658,394,-506,-675,523,730,-790,-109,865,975,-226,651,987,111,862,675,-398,126,-482,457,-24,-356,-795,-575,335,-350,-919,-945,-979,611]
    d = 37
    print(Solution().findTheDistanceValue(arr1, arr2, d))
Create Target Array in the Given Order
class Solution:
    def createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
        target=[]
        for i in range(len(nums)):
            target.insert(index[i],nums[i])
        return target
Find Lucky Integer in an Array
from collections import Counter

class Solution:
    def findLucky(self, arr: List[int]) -> int:
        answer = -1
        cnt = Counter(arr).items()
        
        answer_check = [ item[0] for item in cnt if item[0] == item[1] ]
        
        if answer_check != []:
            answer = max(answer_check)
            
        return answer
Count Largest Group
class Solution:
    def countLargestGroup(self, n: int) -> int:        
        dp={0:0}      
        counts=[0]*(100)       
        for i in range(1,n+1):           
            a=i%10           
            b=i//10
            
            dp[i]  = a+dp[b]
            
            counts[dp[i]]+=1           
        return counts.count(max(counts))
Minimum Value to Get Positive Step by Step Sum
class Solution:
    def minStartValue(self, nums: List[int]) -> int:
        start=1
        
        while True:
            total = start
            is_valid = True
            
            for i in nums:
                total += i
                
                if total < 1:
                    is_valid = False
                    break
                    
            if is_valid:
                return start
            else:
                start+=1
Maximum Score After Splitting a String
class Solution:
    def maxScore(self, s: str) -> int:
        # 왼쪽 0의 갯수 + 오른쪽 1의 갯수  
        # 왼쪽 0의 갯수 + (총 1의 갯수 - 왼쪽 1의 갯수)
        # 총 1의 갯수 + (왼쪽 0의 갯수 - 왼쪽 1의 갯수)
        
        zeros = 0
        ones = 0
        maxScore = float('-inf')
        
        for i in range(len(s)):
            if s[i] == '0':
                zeros += 1
            else:
                ones += 1
            
            if i != len(s) - 1:
                maxScore  = max(maxScore,  zeros - ones)
            
        
        return maxScore + ones
Merge k Sorted Lists
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        link=[]
        i=0
        while len(lists)-1 >= i:
            if lists[i]:
                link.append(lists[i].val)
            else:
                i+=1
                continue
            lists[i]=lists[i].next
            
        link=sorted(link, reverse=True)
        result=None
        for i in link:
            result=ListNode(i, result)
        return result
Sudoku Solver
def checkcorrect(board: List[List[str]], row, col):
    c = set()
    r = set()
    b = set()
    for i in board[row]:
        if i == '.': continue
        if i in r: return False
        r.add(i)
    for j in range(9):
        if board[j][col] == '.': continue
        if board[j][col] in c: return False
        c.add(board[j][col])
    trow = (row // 3) * 3
    tcol = (col // 3) * 3
    for i in range(trow, trow + 3):
        for j in range(tcol, tcol + 3):
            if board[i][j] == '.': continue
            if board[i][j] in b: return False
            b.add(board[i][j])
    return True

def sudoku(board: List[List[str]], empty, n):
    if n >= len(empty): 
        return 1
    for i in range(1, 10):
        board[empty[n][0]][empty[n][1]] = str(i)
        if checkcorrect(board, empty[n][0], empty[n][1]):
            x = sudoku(board, empty, n+1)
            if x == 1: return 1
        board[empty[n][0]][empty[n][1]] = '.'
class Solution:
    def solveSudoku(self, board: List[List[str]]) -> None:
        empty = list()
        for i in range(9):
            for j in range(9):
                if board[i][j] == '.':
                    empty.append([i, j])
        sudoku(board, empty, 0)
N-Queens
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        queen = [['.']*n for _ in range(n)]
        res = []
        C = set() #열 방향
        P = set() #양방향 대각선 = r+c가 같은 위치
        N = set() #음방향 대각선 = r-c가 같은 위치
        def backtracking(row):
            if row == n: #다 둔 경우
                a = [''.join(r) for r in queen]
                res.append(a)
                return
            for c in range(n):
                if c in C: #현재 놓은 자리가 예전에 뒀던 자리의 열 안에 있는가
                    continue
                if row+c in P: #현재 놓은 자리가 예전에 뒀던 자리의 왼 대각선 안에 있는가
                    continue
                if row-c in N: #현재 놓은 자리가 예전에 뒀던 자리의 오른 대각선 안에 있는가
                    continue
                C.add(c)
                P.add(row+c)
                N.add(row-c)
                queen[row][c] = 'Q'
                backtracking(row+1)
                C.remove(c)
                P.remove(row+c)
                N.remove(row-c)
                queen[row][c] = '.'
        backtracking(0)
        return res


Maximal Rectangle
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        n = len(matrix[0])
        height = [0] * (n + 1)
        ans = 0
        for row in matrix:
            for i in range(n):
                height[i] = height[i] + 1 if row[i] == '1' else 0
            stack = [-1]
            for i in range(n + 1):
                while height[i] < height[stack[-1]]:
                    h = height[stack.pop()]
                    w = i - 1 - stack[-1]
                    ans = max(ans, h * w)
                stack.append(i)
        return ans
Rotate Image
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        """
        Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)

        tmp = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                tmp[j][n - 1 - i] = matrix[i][j]
        
        for i in range(n):
            for j in range(n):
                matrix[i][j] = tmp[i][j]

Edit Distance
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0 for _ in range(len(word2)+1)] for _ in range (len(word1)+1)]

        dp[0][0] = 0
        
        for i in range(1, len(word1)+1):
            dp[i][0] = i
            
        for i in range(1, len(word2)+1):
            dp[0][i] = i
            
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]: dp[i][j] = dp[i-1][j-1]
                else: dp[i][j] = min(dp[i-1][j-1] + 1, dp[i-1][j] + 1, dp[i][j-1] + 1)
        
        return dp[len(word1)][len(word2)]
Partition List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        if not head:    return head
        
        smaller, bigger = ListNode(0), ListNode(0)
        s, b = smaller, bigger
        
        if head.val < x:
            s.next = ListNode(head.val)
            s = s.next
        else:
            b.next = ListNode(head.val)
            b = b.next
        
        while head.next:
            head = head.next
            if head.val < x:
                s.next = ListNode(head.val)
                s = s.next
            else:
                b.next = ListNode(head.val)
                b = b.next
        
        s.next = bigger.next
        return smaller.next
Subsets II
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        nums.sort()
        
        def dfs(cur, idx):
            res.append(cur[:])
            
            prev = -11 # -10 <= nums[i] <= 10
            for i in range(idx, len(nums)):
                if prev == nums[i]:
                    continue
                cur.append(nums[i])
                dfs(cur, i + 1)
                cur.pop()
                prev = nums[i]
            
        dfs([], 0)
        return res


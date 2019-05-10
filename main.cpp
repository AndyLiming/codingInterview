#include <algorithm>
#include <bitset>
#include <chrono>
#include <complex>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <ratio>
#include <tuple>
#include <utility>
#include <valarray>

#include <array>
#include <deque>
#include <forward_list>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cmath>
#include <assert.h>

using namespace std;


//structure
struct ListNode {
  int val;
  ListNode * next;
  ListNode(int _val):val(_val),next(NULL) {}
};
struct CompListNode {
  int val;
  CompListNode * next;
  CompListNode *sibling;
  CompListNode(int _val) :val(_val), next(NULL), sibling(NULL) {}
};
//input and output list
ListNode * inputList() {
  int tmp;
  ListNode *helper = new ListNode(-1);
  ListNode *p = helper;
  while (cin >> tmp) {
    //p->next = &ListNode (tmp);
    p->next = new ListNode(tmp);
    p = p->next;
  }
  return helper->next;
}
ListNode * inputList(int n) {
  int tmp;
  ListNode *helper = new ListNode(-1);
  ListNode *p = helper;
  while (n>0) {
    //p->next = &ListNode (tmp);
    cin >> tmp;
    p->next = new ListNode(tmp);
    p = p->next;
    --n;
  }
  return helper->next;
}
void outputList(ListNode *head) {
  while (head != nullptr) {
    cout << head->val << " ";
    head = head->next;
  }
  cout << endl;
}
/*************************************************/
struct TreeNode {
  int val;
  TreeNode* left;
  TreeNode* right;
  TreeNode (int _val):val(_val),left(NULL),right(NULL) {}
};
struct TreeNodeWithP {
  int val;
  TreeNodeWithP* left;
  TreeNodeWithP* right;
  TreeNodeWithP* parent;
  TreeNodeWithP(int _val) :val(_val), left(NULL), right(NULL), parent(NULL) {}
};
/*****************************************************************/

//tools
TreeNode*genTree() {
  TreeNode*root = new TreeNode(1);
  root->left = new TreeNode(2);
  root->right = new TreeNode(3);
  root->left->left = new TreeNode(4);
  root->left->right = new TreeNode(5);
  root->right->right = new TreeNode(6);
  root->left->left->left = new TreeNode(7);
  return root;
}
//TreeNode*genTree(vector<int>&nums) {
//  int len = nums.size();
//  int i = 0;
//  while()
//}

//1 print binary tree by level

void printTreeByLevel(TreeNode* root) {
  if (root == nullptr) return;
  queue<TreeNode*>q;
  TreeNode *last = root;
  TreeNode *nLast = root;
  q.push(root);
  while (!q.empty()) {
    TreeNode *p = q.front();
    cout <<p->val << " ";
    if (p->left != nullptr) q.push(p->left);
    if (p->right != nullptr) q.push(p->right);
    nLast = q.back();
    q.pop();
    if (p == last) {
      cout << endl;
      last = nLast;
    }
  }
  cout << endl;
}

/****************************************************************/

//problems ans solutions

//No.3 重复数字
vector<int> duplicate(vector<int> & nums) {
  vector<int> ans;
  if (nums.size() <= 1) return ans;
  for (int i = 0;i < nums.size();++i) {
    while (nums[i] != i) {
      if (nums[i] == nums[nums[i]]) {
        ans.push_back(nums[i]);
        break;
      }
      int t = nums[i];
      nums[i] = nums[t];
      nums[t] = t;
    }
  }
  return ans;
}

int findOneDump(vector<int> & nums,int start,int end) {
  while (start <= end) {
    int mid = start + (end - start) / 2;
    int count = 0;
    for (auto i : nums) {
      if (i >= start && i <= mid) count++;
    }
    if (start == end) {
      if (count > 1) return start;
      else break;
    }
    if (count > (mid - start + 1)) end = mid;
    else start = mid + 1;
  }
  return -1;
}
/*******************************************************/

//No 4 有序二维数组（左至右递增，上至下递增）查找目标数字是否存在
bool findNumInMatrix(vector<vector<int>> &matrix, int target) {
  int rows = matrix.size();
  int cols = matrix[0].size();
  int r = 0, c = cols-1;
  while (r < rows && c > -1) {
    if (target == matrix[r][c]) return true;
    else if (target > matrix[r][c]) ++r;
    else --c;
  }
  return false;
}
/*****************************************************/

//No 5 替换空格

vector<char> replaceBlank(vector<char> str) {
  int len = str.size(),numBlank = 0;
  for (auto c : str) {
    if (c == ' ') ++numBlank;
  }
  int newLen = len + 2 * numBlank;
  vector<char> ans(len);
  int idOld = len - 1, idNew = newLen - 1;
  while (idOld >= 0 && idNew >= 0) {
    if (str[idOld] == ' ') {
      ans[idNew--] = '0';
      ans[idNew--] = '2';
      ans[idNew--] = '%';
    }
    else {
      ans[idNew--] = str[idOld];
    }
    --idOld;
  }
  return ans;
}

/****************************************************/

//No 6 从尾到头打印链表
void printListReversingly(ListNode * head) {
  if (head == nullptr) return;
  stack<int>s;
  ListNode * p = head;
  while (p) {
    s.push(p->val);
    p = p->next;
  }
  while (!s.empty()) {
    cout << s.top() << " ";
    s.pop();
  }
  cout << endl;
}

void printListReversinglyRec(ListNode * head) {
  if (head != nullptr) {
    if (head->next != nullptr) {
      printListReversinglyRec(head->next);
    }
    cout << head->val << " ";
  }
}
/***************************************************/

//No 7 重建二叉树

TreeNode * rebuildCore(vector<int>&preorder, int startPre, int endPre, vector<int>&inorder, int startIn, int endIn) {
  if (startPre > endPre || startIn > endIn) return nullptr;
  int rootVal = preorder[startPre];
  TreeNode *root = new TreeNode(rootVal);
  int rootIn = startIn;
  while (rootIn<=endIn && inorder[rootIn] != rootVal) ++rootIn;
  root->left = rebuildCore(preorder, startPre + 1, rootIn-startIn+startPre, inorder, startIn, rootIn - 1);
  root->right = rebuildCore(preorder, rootIn - startIn + startPre+1, endPre, inorder, rootIn + 1, endIn);
  return root;
}

TreeNode * rebuildTreeFromPreAndIn(vector<int>&preorder, vector<int>&inorder){
  if (preorder.empty() || inorder.empty() || (preorder.size()!=inorder.size())) return nullptr;
  int len = preorder.size();
  return rebuildCore(preorder, 0, len - 1, inorder, 0, len - 1);
}
/**************************************************************************/

//No 8 二叉树中序遍历的下一个节点
TreeNodeWithP * getNextNode(TreeNodeWithP *node) {
  if (node == nullptr) return nullptr;
  TreeNodeWithP *next = nullptr;
  //node存在右子树，下一个节点是右子树的最左子节点
  if (node->right != nullptr) {
    TreeNodeWithP *right = node->right;
    while (right->left!= nullptr) right = right->left;
    next = right;
  }
  //没有右子树，如果是父节点的左子节点，下一节点是父节点
  //没有右子树，是父节点的右节点，下一节点是沿父节点上溯，第一个是其父节点左子节点的节点
  else if (node->parent != nullptr) {
    TreeNodeWithP *current = node;
    TreeNodeWithP *parent = node->parent;
    while (parent != nullptr && current == parent->right) {
      current = parent;
      parent = parent->parent;
    }
    next = parent;
  }
  return next;
}
/***********************************************************************/

//No 9 两个栈实现一个队列
template<typename T>
class myQueue {
public:
  void myQpush(const T &t);
  T myQpop();
private:
  stack<T> s1;
  stack<T> s2;
};
template<typename T>
void myQueue<T>::myQpush(const T &t) {
  s1.push(t);
}
template<typename T>
T myQueue<T>::myQpop() {
  if (s2.empty()) {
    while (!s1.empty()) {
      T &data = s1.top();
      s1.pop();
      s2.push(data);
    }
  }
  if (s2.empty()) throw new exception("queue empty!");
  T ans = s2.top();
  s2.pop();
  return ans;
}
//两个队列实现一个栈
template<typename T>
class myStack {
public:
  void mySpush(const T &t);
  T mySpop();
private:
  queue<T> q1;
  queue<T> q2;
};

template<typename T>
void myStack<T>::mySpush(const T&t) {
   if(!q1.empty()) q1.push(t);
   else q2.push(t);
}

template<typename T>
T myStack<T>::mySpop() {
  if (q1.empty()) {
    while (q1.size() > 1) {
      T &tmp = q1.front();
      q1.pop();
      q2.push(tmp);
    }
    T ans = q1.top();
    q1.pop();
  }
  else if (q2.empty()) {
    while (q2.size() > 1) {
      T &tmp = q2.front();
      q2.pop();
      q1.push(tmp);
    }
    T ans = q2.top();
    q1.pop();
  }
  return ans;
}
/*************************************************************/

//No.10 斐波那契数列
//DP
long long fibonacci(int n) {
  vector<long long> fib{ 0,1 };
  if (n < 2) return fib[n];
  long long fibN = 0;
  for (int i = 2;i <= n;++i) {
    fibN = fib[0] + fib[1];
    fib[0] = fib[1];
    fib[1] = fibN;
  }
  return fibN;
}
/**********************************************************/

//No.11 旋转数组的最小数字
//binary search
int minNumInRotateArray(vector<int> &nums) {
  if (nums.empty()) throw new exception("invalid input!");
  if (nums.size() == 1) return nums[0];
  int low = 0, high = nums.size()-1,mid = 0;
  while (nums[low]>=nums[high]) {
    if ((high - low) == 1) {
      mid = high;
      break;
    }
    mid = low + (high - low) / 2;
    if (nums[low] == nums[high] && nums[low] == nums[mid]) {
      //只能顺序查找最小值
      int ans = nums[low];
      for (int i = low + 1;i <= high;++i) {
        ans = min(ans, nums[i]);
      }
      return ans;
    }
    if (nums[mid] >= nums[low]) low = mid;
    else if(nums[mid]<=nums[high]) high = mid;
  }
  return nums[mid];
}
/**********************************************************/

//No.12 矩阵中的路径

bool hasPathCore(vector<vector<char>> &matrix, string s, int pathLen, vector<vector<bool>>&visited,int r,int c) {
  if (pathLen == s.size()) return true;
  bool ans = false;
  if (r >= 0 && r < matrix.size() && c >= 0 && c < matrix[0].size() && !visited[r][c] && matrix[r][c] == s[pathLen]) {
    ++pathLen;
    visited[r][c] = true;
    ans = hasPathCore(matrix, s, pathLen, visited, r + 1, c)
      || hasPathCore(matrix, s, pathLen, visited, r - 1, c)
      || hasPathCore(matrix, s, pathLen, visited, r, c + 1)
      || hasPathCore(matrix, s, pathLen, visited, r, c - 1);
    if (ans == false) {
      --pathLen;
      visited[r][c] = false;
    }
  }
  return ans;
}

bool hasPath(vector<vector<char>> &matrix, string s) {
  if (matrix.empty() || matrix[0].empty() || s.empty()) return false;
  int rows = matrix.size(), cols = matrix[0].size();
  vector<vector<bool>> visited(rows, vector<bool>(cols, false));
  int pathLen = 0;
  for (int i = 0;i < rows;++i) {
    for (int j = 0;j < cols;++j) {
      if (hasPathCore(matrix, s, pathLen, visited, i, j)) return true;
    }
  }
  return false;
}
/********************************************************/

//No.13 机器人运动范围

int getdigitSum(int n) {
  int k = 0;
  while (n) {
    k += (n % 10);
    n /= 10;
  }
  return k;
}
bool canArrived(int th, int r, int c, int rows,int cols,vector<vector<bool>>& visited) {
  if (r >= 0 && r < rows&&c >= 0 && c < cols && visited[r][c] == true && (getdigitSum(r) + getdigitSum(c) <= th)) return true;
  else return false;
}
int movingCountCore(int th,int r,int c,int rows,int cols,vector<vector<bool>>&visited){
  int cnt = 0;
  if (canArrived(th, r, c, rows, cols, visited)) {
    visited[r][c] = true;
    cnt = 1 + movingCountCore(th, r + 1, c, rows, cols, visited) + movingCountCore(th, r - 1, c, rows, cols, visited)
      + movingCountCore(th, r, c + 1, rows, cols, visited) + movingCountCore(th, r, c - 1, rows, cols, visited);
  }
  return cnt;
}
int movingCount(int th, int rows, int cols) {
  if (th < 0 || rows <= 0 || cols <= 0) return 0;
  vector<vector<bool>> visited(rows, vector<bool>(cols, false));
  return movingCountCore(th, 0, 0, rows, cols, visited);
}
/**********************************************************/

//No.14 剪绳子 各小段长度乘积最大

int maxProductCut(int length) {
  if (length < 2) return 0;
  if (length == 2) return 1;
  if (length == 3) return 2;
  int timesOf3 = length / 3;
  if ((length - timesOf3 * 3) == 1) --timesOf3;
  int timesOf2 = (length - timesOf3 * 3) / 2;
  return (int)pow(3, timesOf3)*(int)pow(2, timesOf2);
}
/***********************************************************/

//No.15 二进制中1的个数
int numOf1(int n) {
  int cnt = 0;
  while (n) {
    n &= (n - 1);
    ++cnt;
  }
  return cnt;
}
bool is2power(int n) {
  if (n&(n - 1) == 0) return true;
  else return false;
}
int digitsMtoN(int m, int n) {
  int k = m^n;
  return numOf1(k);
}
/**********************************************************/

//No.16

double myPow(double base, int exp) {
  if (exp == 0) return 1;
  if (exp == 1) return 0;
  double ans = myPow(base, exp >> 1);
  ans *= ans;
  if (exp&0x1 == 1) ans *= base;//位运算更高效
  return exp > 0 ? ans : 1 / ans;
}

/*********************************************************/
//No.17 打印从1到最大的n位数
void printNum(string &num) {
  int i = 0;
  while (i < num.size() && num[0] == '0')++i;
  cout << num.substr(i) << endl;
}
void print1ToMaxOfNDigitsRec(string &num, int len, int index) {
  if (index == len - 1) {
    printNum(num);
    return;
  }
  for (int i = 0;i <= 9;++i) {
    num[index+1] = i + '0';
    print1ToMaxOfNDigitsRec(num, len, index + 1);
  }
}
void print1ToMaxOfNDigits(int n) {
  if (n <= 0) return;
  string num(n,'0');
  for (int i = 0;i <= 9;++i) {
    num[0] = i + '0';
    print1ToMaxOfNDigitsRec(num, n, 0);
  }
}
/***********************************************************/

//No.18 删除节点
ListNode* deleteNode(ListNode * head,ListNode * toBeDelete) {
  if (!head || !toBeDelete) return head;
  if (head == toBeDelete) return nullptr;
  else if (toBeDelete->next == nullptr) {//尾节点
    ListNode *p = head;
    while (p->next != toBeDelete)p = p->next;
    delete toBeDelete;
    p->next = nullptr;
  }
  else {
    ListNode *t = toBeDelete->next;
    toBeDelete->val = t->val;
    toBeDelete->next = t->next;
    delete t;
  }
  return head;
}

/**********************************************************/

//No.19 正则表达式匹配
bool matchReCore(char * sp, char * pp) {
  if (*sp == '\0' && *pp == '\0') return true;
  if (*sp != '\0' && *pp == '\0') return false;
  if (*(pp + 1) == '*') {
    if (*pp == *sp || (*pp == '.'&&*sp != '\0')) {
      return matchReCore(sp + 1, pp + 2) || matchReCore(sp + 1, pp) || matchReCore(sp, pp + 2);
    }
    else {
      return matchReCore(sp, pp + 2);
    }
  }
  if (*sp == *pp || (*pp == '.'&&sp != '\0')) return matchReCore(sp + 1, pp + 1);
  return false;
}
bool matchRe(char * str, char * pattern) {
  if (str == nullptr || pattern== nullptr) return false;
  return matchReCore(str,pattern);
}
/***********************************************************************/

//No.20 表示数值的字符串 能否表示成为：A[.[B]][e|EC] .B[e|EC] 的形式
//首先检测A整数部分，遇到.检测小数部分，遇到e或E检测指数部分
bool scanUnsignedint(const char **str) {
  const char *start = *str;
  while (**str != '\0' && **str >= '0' && **str <= '9')++(*str);
  return *str > start;
}
bool scanInt(const char **str) {
  if (**str == '+' || **str == '-') ++(*str);
  return scanUnsignedint(str);
}
bool isNumeric(const char *str) {
  if (str == nullptr) return false;
  bool ans = scanInt(&str);//A
  if (*str == '.') {
    ++str;
    ans  = scanUnsignedint(&str) || ans;//B
  }
  if (*str == 'e' || *str == 'E') {
    ++str;
    ans = ans && scanInt(&str);
  }
  return ans && *str == '\0';
}

//bool scanUnsignedint(string::iterator * sit) {
//  string::iterator start = *sit;
//  while (isdigit(**sit)) ++sit;
//  return *sit != start;
//}
//bool scanInt(string::iterator * sit) {
//  if (**sit == '+' || **sit == '-') ++sit;
//  return scanUnsignedint(sit);
//}
//bool isNumeric(string str) {
//  int i = 0;
//  while (i < str.size() && str[i] == ' ')++i;
//  int j = i;
//  while (j < str.size() && j != ' ')++j;
//  str = str.substr(i, j - i);
//  cout << str << endl;
//  if (str.empty()) return false;
//  string::iterator sit = str.begin();
//  bool ans = scanInt(&sit);
//  if (*sit == '.') {
//    ++sit;
//    ans = scanUnsignedint(&sit) || ans;//B
//  }
//  if (*sit == 'e' || *sit == 'E') {
//    ++sit;
//    ans = ans && scanInt(&sit);
//  }
//  return ans && sit==str.end();
//}
bool isNumeric2(string s) {
  int len = s.size();
  int left = 0, right = len - 1;
  bool digitExisited = false, dotExisted = false, eExisted = false;
  //remove blanks in the head and tail
  while (s[left] == ' ')++left;
  while (s[right] == ' ')--right;

  //has only one character but not a number
  if (left >= right && (s[left] < '0' || s[left] > '9')) return false;
  
  if (s[left] == '.') dotExisted = true;//only has decimal part
  else if (s[left] >= '0' && s[left] <= '9') digitExisited = true;//has non-signed integer part
  else if (s[left] != '+' && s[left] != '-') return false;

  for(int i =left+1;i<right;++i){
    if (s[i] >= '0' && s[i] <= '9') digitExisited = true;
    else if (s[i] == 'e' || s[i] == 'E') { // e/E cannot follow +/-, must follow a digit
      if (!eExisted && s[i - 1] != '+' && s[i - 1] != '-' && digitExisited) eExisted = true;
      else return false;
    }
    else if (s[i] == '+' || s[i] == '-') { // +/- can only follow e/E
      if (s[i - 1] != 'e' && s[i - 1] != 'E') return false;
    }
    else if (s[i] == '.') { // dot can only occur once and cannot occur after e/E
      if (!dotExisted && !eExisted) dotExisted = true;
      else return false;
    }
    else return false;
  }
  if (s[right] >= '0' && s[right] <= '9') return true;
  else if (s[right] == '.' && !dotExisted && !eExisted && digitExisited) return true;
  else return false;

}
/***********************************************************************/

//No.21 调整数组使奇数位于偶数之前
bool isEven(int n) {//compare condition
  return (n & 1) == 0;
}
void reOrder(vector<int>&nums) {
  if (nums.size() <= 1) return;
  int left = 0, right = nums.size() - 1;
  while (left < right) {
    while (left < right && !isEven(nums[left])) ++left;
    while (left < right&&isEven(nums[right]))--right;
    if (left < right) {
      swap(nums[left], nums[right]);
    }
  }
}
/**************************************************************************/
//No.22 链表中倒数第k个节点
ListNode * KthNodeToTail(ListNode * head, int k) {
  if (head == nullptr) return nullptr;
  ListNode *p1 = head;
  for (int i = 0;i < k-1;++i) {//p1 move k-1 not k!
    if(p1->next!=nullptr) p1 = p1->next;
    else return nullptr;
  }
  ListNode *p2 = head;
  while (p1->next != nullptr) {
    p1 = p1->next;
    p2 = p2->next;
  }
  return p2;
}
/**************************************************************************/
//No.23 链表环的入口
ListNode *meetNode(ListNode * head) {
  if (head == nullptr) return nullptr;
  //has a loop or not
  ListNode *fast = head, *slow = head;
  while (fast != nullptr && fast->next != nullptr) {
    if (fast == slow) return slow;
    fast = fast->next->next;
    slow = slow->next;
  }
  return nullptr;
}
ListNode * entryNodeOfLoop(ListNode * head) {
  ListNode *meetN = meetNode(head);
  if (meetN == nullptr) return nullptr;
  ListNode * p1 = head, *p2 = meetN;
  while (p1 != p2) {
    p1 = p1->next;
    p2 = p2->next;
  }
  return p1;
}
ListNode *entryNodeOfLoop2(ListNode*head) {
  ListNode * p1 = head, *p2 = head;
  while (p1 != nullptr &&p1->next != nullptr) {
    p1 = p1->next->next;
    p2 = p2->next;
    if (p1 == p2) {
      p1 = head;
      while (p1 != p2) {
        p1 = p1->next;
        p2 = p2->next;
      }
      return p1;
    }
  }
  return nullptr;
}
/**************************************************************************/
//No.24 反转链表
ListNode* reverseList(ListNode *head) {
  if (head == nullptr || head->next == nullptr) return head;
  ListNode *pre = head, *cur = head->next;
  pre->next = nullptr;
  while (cur->next != nullptr) {
    ListNode *t = cur->next;
    cur->next = pre;
    pre = cur;
    cur = t;
  }
  cur->next = pre;
  return cur;
}
/**************************************************************************/
//No.25 合并两个有序链表
ListNode *mergeSortedList(ListNode *head1, ListNode *head2) {
  if (head1 == nullptr) return head2;
  else if (head2 == nullptr) return head1;
  ListNode *mergedHead=nullptr;
  if (head1->val < head2->val) {
    mergedHead = head1;
    mergedHead->next = mergeSortedList(head1->next, head2);
  }
  else {
    mergedHead = head2;
    mergedHead->next = mergeSortedList(head1, head2->next);
  }
  return mergedHead;
}
/**************************************************************************/
//No.26 树的子结构
bool treeAhasTreeB(TreeNode *rootA, TreeNode *rootB) {
  if (rootB == nullptr) return true;
  if (rootA == nullptr) return false;
  if (rootA->val != rootB->val) return false;
  return treeAhasTreeB(rootA->left, rootB->left) && treeAhasTreeB(rootA->right, rootB->right);
}
bool hasSubTree(TreeNode *rootA, TreeNode *rootB) {
  bool result = false;
  if (rootA->val == rootB->val) treeAhasTreeB(rootA, rootB);
  if (!result) hasSubTree(rootA->left, rootB);
  if (!result) hasSubTree(rootA->right, rootB);
  return result;
}
/**************************************************************************/
//No.27 二叉树的镜像（水平翻转）
void mirrorTree(TreeNode *root) {
  if (root == nullptr) return;
  if (root->left == nullptr && root->right == nullptr) return;
  TreeNode *tmp = root->left;
  root->left = root->right;
  root->right = tmp;
  if (root->left != nullptr) mirrorTree(root->left);
  if (root->right != nullptr) mirrorTree(root->right);
}
/*************************************************************************/
//No.28 对称的二叉树
bool isSymmetrical(TreeNode *root1, TreeNode *root2) {
  if (root1 == nullptr && root2 == nullptr) return true;
  if (root1 == nullptr || root2 == nullptr) return false;
  if (root1->val != root2->val) return false;
  return isSymmetrical(root1->left, root2->right) && isSymmetrical(root1->right, root2->left);
}
bool isSymTree(TreeNode*root) {
  return isSymmetrical(root, root);
}
/*************************************************************************/
//No.29 顺时针打印矩阵
void printInCircle(vector<vector<int>>&matrix,int rows, int cols,int start) {
  int endX = cols - 1 - start;
  int endY = rows - 1 - start;
  for (int i = start;i <= endX;++i) cout << matrix[start][i]<<" ";
  for (int i = start + 1;i <= endY;++i)cout << matrix[i][endX] << " ";
  for (int i = endX - 1;i >= start;--i)cout << matrix[endY][i] << " ";
  for (int i = endY - 1;i > start;--i)cout << matrix[i][start] << " ";
}
void printMatClkWise(vector<vector<int>>&matrix) {
  if (matrix.empty() || matrix[0].empty()) return;
  int rows = matrix.size(), cols = matrix[0].size();
  int start = 0;
  while (cols > 2 * start&&rows > 2*start) {
    printInCircle(matrix, rows, cols, start);
    ++start;
  }
}
/*************************************************************************/
//No.30 包含min函数（或max）的栈
template<typename T>
class stackWithMin {
public:
  void push(const T&t);
  T pop();
  T getMin();
private:
  stack<T> data;
  stack<T> minD;
};

template<typename T>
void stackWithMin<T>::push(const T&t) {
  data.push(t);
  if (minD.empty() || t < minD.top()) minD.push(t);
  else minD.push(minD.top());
}
template<typename T>
T stackWithMin<T>::pop() {
  assert(!data.empty() && !minD.empty());
  T t = data.top();
  data.pop();
  minD.pop();
  return t;
}
template<typename T>
T stackWithMin<T>::getMin() {
  assert(!data.empty() && !minD.empty());
  return minD.top();
}
/*************************************************************************/
//No.31 栈的压入 弹出序列
bool isPopOrder(vector<int>&n1, vector<int>&n2) {
  if (n1.size() != n2.size()) return false;
  int i = 0, j = 0;
  stack<int> s;
  while (j < n2.size()) {
    while (s.empty() || s.top() != n2[j]) {
      if (i == n1.size())break;
      s.push(n1[i]);
      ++i;
    }
    if (s.top() != n2[j])break;
    s.pop();
    ++j;
  }
  if (s.empty() && j == n2.size()) return true;
  else return false;
}
/*************************************************************************/
//No.32 二叉树的层序遍历
//1.不分行的遍历
void levelPrintTree(TreeNode *root) {
  if (root == nullptr) return;
  queue<TreeNode*>q;
  q.push(root);
  while (!q.empty()) {
    TreeNode *t = q.front();
    q.pop();
    cout << t->val << " ";
    if (t->left) q.push(t->left);
    if (t->right) q.push(t->right);
  }
  cout << endl;
}
//2.分行的遍历
void levelPrintTreeRows(TreeNode*root) {
  if (root == nullptr) return;
  TreeNode*last = root;
  TreeNode*nLast = root;
  queue<TreeNode*>q;
  q.push(root);
  while (!q.empty()) {
    TreeNode *t = q.front();
    
    if (t->left!= nullptr) q.push(t->left);
    if (t->right!= nullptr) q.push(t->right);
    nLast = q.back();
    q.pop();
    cout << t->val;
    if (t == last) {
      cout << endl;
      last = nLast;
    }
    else cout << " ";
  }
}
//3.之字形遍历
void levelPrintTreeZigzag(TreeNode*root) {
  if (root == nullptr) return;
  stack<TreeNode*>levels[2];
  int cur = 0, next = 1;
  levels[cur].push(root);
  while (!levels[0].empty() || !levels[1].empty()) {
    TreeNode *t = levels[cur].top();
    levels[cur].pop();
    cout << t->val;
    if (cur == 0) {
      if (t->left != nullptr) levels[next].push(t->left);
      if (t->right != nullptr) levels[next].push(t->right);
    }
    else {
      if (t->right != nullptr) levels[next].push(t->right);
      if (t->left != nullptr) levels[next].push(t->left);
    }
    if (levels[cur].empty()) {
      cout << endl;
      cur = 1 - cur;
      next = 1 - next;
    }
    else cout << " ";
  }
}
/*************************************************************************/
//No.33 二叉搜索树的后序遍历序列
bool isPostOrderOfBST(vector<int>&nums, int start, int end) {
  if (start > end || nums.empty()) return false;
  int root = nums[end];
  int i = start;
  while (i < end && nums[i] < root) ++i;//大于root的第一个，右子树的开始
  for (int j = i;j < end;++j) {
    if (nums[j] < root) return false;
  }
  bool left = true,right=true;
  if (i > start) left = isPostOrderOfBST(nums, start, i-1);
  if (i < end) right = isPostOrderOfBST(nums, i, end-1);
  return left&&right;
}
bool isPostOrderOfBST(vector<int>&nums) {
  if (nums.empty()) return false;
  return isPostOrderOfBST(nums, 0, nums.size() - 1);
}
/*************************************************************************/
//No.34 二叉树中和为某一值的路径
void findPath(TreeNode*root, vector<int>&path, int sum, int curSum) {
  curSum += root->val;
  path.push_back(root->val);
  bool isLeaf = root->left == nullptr&&root->right == nullptr;
  if (curSum == sum && isLeaf) {
    cout << "find a path!" << endl;
    for (auto n : path) cout << n << " ";
    cout << endl;
  }
  if (root->left != nullptr) findPath(root->left, path, sum, curSum);
  if (root->right != nullptr) findPath(root->right, path, sum, curSum);
  path.pop_back();
}
void findPath(TreeNode*root, int sum) {
  if (root == nullptr) return;
  vector<int>path;
  findPath(root, path, sum, 0);
}
/*************************************************************************/
//No 35 复杂链表的复制
void cloneListNodes(CompListNode*head) {
  CompListNode*p = head;
  while (p != nullptr) {
    CompListNode*newNode = new CompListNode(p->val);
    newNode->next = p->next;
    p->next = newNode;
    p = newNode->next;
  }
}
void connectSiblings(CompListNode*head) {
  CompListNode*p = head;
  while (p != nullptr) {
    CompListNode*pClone = p->next;
    if (p->sibling != nullptr) {
      pClone->sibling = p->sibling->next;
    }
    p = pClone->next;
  }
}
CompListNode*reconNodes(CompListNode*head) {
  CompListNode *p = head, *pClone = nullptr, *pCloneHead = nullptr;
  if (p != nullptr) {
    pClone = pCloneHead = p->next;
    p->next = pClone->next;
    p = pClone->next;
  }
  while (p != nullptr) {
    pClone->next = p->next;
    pClone = pClone->next;
    p->next = pClone->next;
    p = p->next;
  }
  return pCloneHead;
}
CompListNode*cloneCompList(CompListNode*head) {
  if (head == nullptr) return head;
  cloneListNodes(head);
  connectSiblings(head);
  return reconNodes(head);
}
/*************************************************************************/
//No 36 二叉搜索树与双向链表(in order)
void convertNode(TreeNode*node, TreeNode**pLastNodeInList) {
  if (node == nullptr)return;
  TreeNode*current = node;
  if (current->left != nullptr) convertNode(current->left, pLastNodeInList);
  current->left = *pLastNodeInList;
  if (*pLastNodeInList != nullptr) (*pLastNodeInList)->right = current;
  *pLastNodeInList = current;
  if(current->right!=nullptr) convertNode(current->right, pLastNodeInList);
}
TreeNode *convert(TreeNode*root) {
  TreeNode*pLastNodeInList = nullptr;
  convertNode(root, &pLastNodeInList);
  TreeNode*headOfList = pLastNodeInList;
  while (headOfList != nullptr&&headOfList->left != nullptr) headOfList = headOfList->left;
  return headOfList;
}

/*************************************************************************/
//No 37 序列化（及反序列化）二叉树
void serialize(TreeNode*root,ostream&stream) {
  if (root == nullptr) {
    stream << "$,";
    return;
  }
  stream << root->val << ",";
  serialize(root->left, stream);
  serialize(root->right, stream);
}
void deSerialize(TreeNode**proot, istream&stream) {
  char c;
  stream >> c;
  if (isdigit(c)) {
    *proot = new TreeNode(c - '0');
    deSerialize(&((*proot)->left), stream);
    deSerialize(&((*proot)->right), stream);
  }
}
/*************************************************************************/
//No 38 字符串的排列
void permutationRec(string s, int index) {
  if (index == s.size()) cout << s<<endl;
  else {
    for (int i = index;i < s.size();++i) {
      char tmp = s[i];
      s[i] = s[index];
      s[index] = tmp;
      permutationRec(s, index + 1);
      tmp = s[i];
      s[i] = s[index];
      s[index] = tmp;
    }
  }
}
void permutation(string s) {
  if (s.empty()) return;
  permutationRec(s, 0);
}
//8queens
bool check(vector<int>&colIds,int k) {
  if (k == 0)return true;
  for (int i = 0;i < k;++i) {
    if (colIds[i] == colIds[k] || abs(colIds[i] - colIds[k]) == k - i) return false;
  }
  return true;
}
void eightQueens() {
  vector<int>colIds(8);
  int count = 0;
  for (colIds[0] = 0;colIds[0] < 8;++colIds[0]) {
    for (colIds[1] = 0;colIds[1] < 8;++colIds[1]) {
      if (!check(colIds, 1)) continue;
      for (colIds[2] = 0;colIds[2] < 8;++colIds[2]) {
        if (!check(colIds, 2)) continue;
        for (colIds[3] = 0;colIds[3] < 8;++colIds[3]) {
          if (!check(colIds, 3)) continue;
          for (colIds[4] = 0;colIds[4] < 8;++colIds[4]) {
            if (!check(colIds, 4)) continue;
            for (colIds[5] = 0;colIds[5] < 8;++colIds[5]) {
              if (!check(colIds, 5)) continue;
              for (colIds[6] = 0;colIds[6] < 8;++colIds[6]) {
                if (!check(colIds, 6)) continue;
                for (colIds[7] = 0;colIds[7] < 8;++colIds[7]) {
                  if (!check(colIds, 7)) continue;
                  for (auto n : colIds) cout << n << " ";
                  cout << endl;
                  ++count;
                }
              }
            }
          }
        }
      }
    }
  }
  cout << "total num: " << count << endl;
}
//int cnt;
//void backTrace(int k,int n,vector<int>&colIds) {
//  if (k > n) {
//    for (int i = 0;i < 8;++i) cout << colIds[i] << " ";
//    cout << endl;
//    ++cnt;
//  }
//  else {
//    for (int i = 0;i < 8;++i) {
//      colIds[k] = i;
//      if (check(colIds, k)) {
//        //cout << "b" << endl;
//        backTrace(k + 1, n, colIds);
//      }
//    }
//  }
//}
//void eightQueens2() {
//  vector<int>colIds(8);
//  cnt = 0;
//  backTrace(0, 7, colIds);
//  cout << cnt << endl;
//}
/*************************************************************************/
//No 39 出现次数过半的数字
int moreThanHalfNum(vector<int>&nums) {
  if (nums.empty()) throw new exception("invalid input array");
  int res = nums[0];
  int times = 1;
  for (int i = 0;i < nums.size();++i) {
    if (times == 0) {
      res = nums[i];
      times = 1;
    }
    else if (res == nums[i])++times;
    else --times;
  }
  int times2 = 0;
  for (auto n : nums) {
    if (res == n)++times2;
  }
  if (times2 * 2 <= nums.size()) throw new exception("ther is no such number that shows up more than half");
  else return res;
}
/*************************************************************************/
//No 40 最小的K个数
int myPartition(vector<int>&nums, int start, int end) {
  int i = start, j = end, pivot = nums[start];
  while (i < j) {
    while (i<j&&nums[j]>pivot)--j;
    swap(nums[i], nums[j]);
    while (i < j&&nums[i] <= pivot)++i;
    swap(nums[i], nums[j]);
  }
  return i;
}
vector<int> getLeastKNum(vector<int>&nums, int k) {
  if (nums.empty() || k > nums.size()) return vector<int>();
  int start = 0, end = nums.size() - 1;
  int index = myPartition(nums, start, end);
  while (index != k - 1) {
    if (index > k - 1) {
      index = myPartition(nums, start, index - 1);
    }
    else {
      index = myPartition(nums, index + 1, end);
    }
  }
  vector<int>ans(nums.begin(),nums.begin()+k);
  return ans;
}
vector<int> getLeastKNum2(vector<int>&nums, int k) {
  priority_queue<int> heap;
  for (auto n : nums) {
    heap.push(n);
    if (heap.size() > k) heap.pop();
  }
  vector<int>ans;
  while (!heap.empty()) {
    ans.push_back(heap.top());
    heap.pop();
  }
  return ans;
}
/*************************************************************************/
//No 41 数据流中的中位数
template<typename T>
class dynamicArray {
public:
  void insert(T num) {
    if ((minimal.size() + maximum.size()) & 1 == 0) {//偶数
      if (maximin.size() > 0 && num < maximum[0]) {
        maximum.push_back(num);
        push_heap(maximum.begin(), maximum.end(), less<T>);
        num = maximum[0];
        pop_heap(maximum.begin(), maximum.end(), less<T>);
        maximum.pop_back();
      }
      minimal.push_back(num);
      push_heap(minimal.begin(), minimal.end(), greater<T>);
    }
    else {
      if (minimal.size() > 0 && num < minimal[0]) {
        minimal.push_back(num);
        push_heap(minimal.begin(), minimal.end(), greater<T>);
        num = minimal[0];
        push_heap(minimal.begin(), minimal.end(), greater<T>);
        minimal.pop_back();
      }
      maximum.push_back(num);
      push_heap(maximum.begin(), maximum.end(), less<T>);
    }
  }
  T getMid() {
    int size = minimal.size() + maximum.size();
    if (size == 0) throw exception("no data is available!");
    T mid = 0;
    if ((size & 1) == 1)mid = minimal[0];
    else mid = (minimal[0] + maximum[0]) / 2;
    return T;
  }
private:
  vector<T>minimal;
  vector<T>maximum;
};
/*************************************************************************/
//No 42 连续子数组的最大和
int maxSumOfSubArray(vector<int> &nums) {
  if (nums.empty()) return 0;
  if (nums.size() == 1) return nums[0];
  vector<int>dp(nums.size());
  dp[0] = nums[0];
  int maxSum = dp[0];
  for (int i = 1;i < nums.size();++i) {
    dp[i] = dp[i - 1] > 0 ? dp[i - 1] + nums[i] : nums[i];
    maxSum = max(maxSum, dp[i]);
  }
  return maxSum;
}
/**************************************************************************/
//No 43 1-N中1出现的个数
int numOf1Between1AndN(int n) {
  if (n <= 0) return 0;
  int i = 1,high = n,cnt=0;
  while (high != 0) {
    high = n / pow(10, i);
    int temp = n / pow(10, i - 1);
    int cur = temp % 10;//cur表示第i位上的值，从1开始计算
    int low = n - temp * pow(10, i - 1);//low表示当前位的低位
    if (cur < 1){
      cnt += high * pow(10, i - 1);
    }
    else if (cur > 1){
      cnt += (high + 1) * pow(10, i - 1);
    }
    else{
      cnt += high * pow(10, i - 1);
      cnt += (low + 1);
    }
    i++;
  }
  return cnt;
}
/**************************************************************************/
//No 44 数字序列中的某一位数字
int nthDigit(int n) {
  assert(n >= 0);
  long long digit = 1, start = 1;;
  long long num = 9;
  while (n > num*digit) {
    n -= num*digit;
    ++digit;
    num *= 10;
    start *= 10;
  }
  start += (n - 1) / digit;
  //cout << start << endl;
  return to_string(start)[(n - 1) % digit] - '0';
}
/**************************************************************************/
//No 45 smallest string (num)
string smallestStr(vector<string>& vs) {
  sort(vs.begin(), vs.end(), [](string s1, string s2) {return s1 + s2 < s2 + s1;});
  string ans = "";
  for (auto s : vs) ans += s;
  return ans;
}
/**********************************************************************/
//No 46 把数字翻译成字符串
int getTranslationCount(int n) {
  if (n < 0) return 0;
  string number = to_string(n);
  int len = number.size();
  vector<int>dp(len);
  dp[len - 1] = 1;
  //int count = 0;
  for (int i = len - 2;i >= 0;--i) {
    dp[i] = dp[i + 1];
    int t = (number[i] - '0') * 10 + (number[i + 1] - '0');
    if (t >= 10 && t <= 25) {
      if(i + 2 < len) dp[i] += dp[i + 2];
      else dp[i] += 1;
    }
  }
  return dp[0];
}
/**********************************************************************/
//No 47 礼物的最大价值
int maxGiftValue(vector<vector<int>>&matrix) {
  if (matrix.empty() || matrix[0].empty()) return 0;
  int rows = matrix.size(), cols = matrix[0].size();
  vector<vector<int>> dp(rows, vector<int>(cols, 0));
  dp[0][0] = matrix[0][0];
  for (int i = 1;i < cols;++i) dp[0][i] = dp[0][i - 1] + matrix[0][i];
  for (int i = 1;i < rows;++i) dp[i][0] = dp[i-1][0] + matrix[i][0];
  for (int i = 1;i < rows;++i) {
    for (int j = 1;j < cols;++j) {
      dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]) + matrix[i][j];
    }
  }
  return dp[rows - 1][cols - 1];
}
/**********************************************************************/
//No 48 最长不含重复字符的子字符串
int longestNoDup(string s) {
  vector<int>pos(26, -1);
  int maxLen = 0,curLen=0;
  for (int i = 0;i < s.size();++i) {
    int preId = pos[s[i] - 'a'];
    if (preId < 0 || i - preId > curLen) ++curLen;
    else {
      curLen = i - preId;
    }
    maxLen = max(maxLen, curLen);
    pos[s[i] - 'a'] = i;
  }
  return maxLen;
}
/**********************************************************************/
//No 49 丑数
bool isUgly(int num)
{
  if (num <= 0) return false;
  if (num == 1) return true;
  while (num > 1) {
    if (num % 2 != 0 && num % 3 != 0 && num % 5 != 0) return false;
    else {
      if (num % 2 == 0) num /= 2;
      if (num % 3 == 0) num /= 3;
      if (num % 5 == 0) num /= 5;
    }
  }
  return true;
}
int nthUglyNumber(int n)
{
  if (n <= 1) return 1;
  int t2 = 0, t3 = 0, t5 = 0;
  vector<int> dp(n);
  dp[0] = 1;
  for (int i = 1;i < n;++i) {
    dp[i] = min(dp[t2] * 2, min(dp[t3] * 3, dp[t5] * 5));
    if (dp[i] == dp[t2] * 2) t2++;
    if (dp[i] == dp[t3] * 3) t3++;
    if (dp[i] == dp[t5] * 5) t5++;
  }
  return dp[n - 1];
}
/**********************************************************************/
//No 50 first not repeating char in a string
char firstNotRepeat(string s) {
  if (s.empty()) return '\0';
  vector<int>charMap(256, 0);
  for (auto c : s) ++charMap[c];
  for (auto c : s) {
    if (charMap[c] == 1) return c;
  }
  return '\0';
}
/**********************************************************************/
//No 51 逆序对
int mergeRec(vector<int>& nums, int start, int end, vector<int>& helper) {
  if (start >= end) return 0;
  int count = 0;
  int mid = (start + end) / 2;
  int start1 = start, end1 = mid;
  int start2 = mid + 1, end2 = end;
  int left = mergeRec(nums, start1, end1, helper);
  int right = mergeRec(nums, start2, end2, helper);
  int k = start;
  while (start1 <= end1&&start2 <= end2) {
    if (nums[start1] <= nums[start2]) helper[k++] = nums[start1++];
    else{
      helper[k++] = nums[start2++];
      count += end1 - start1 + 1;
    }
    //helper[k++] = nums[start1] < nums[start2] ? nums[start1++] : nums[start2++];
  }
  while (start1 <= end1) helper[k++] = nums[start1++];
  while (start2 <= end2) helper[k++] = nums[start2++];
  for (k = start;k <= end;++k) {
    nums[k] = helper[k];
  }
  return count + left + right;
}
int inversePairs(vector<int>&nums) {
  if (nums.size() <= 1)return 0;
  int len = nums.size();
  vector<int>helper(nums.begin(), nums.end());
  return mergeRec(nums, 0, len - 1, helper);
}
/**********************************************************************/
//No 52 两个链表第一个公共节点
int getListLength(ListNode*head) {
  if (head == nullptr) return 0;
  ListNode*p = head;
  int l = 0;
  while (p != nullptr) {
    p = p->next;
    ++l;
  }
  return l;
}
ListNode*firstComNode(ListNode*head1, ListNode*head2) {
  int len1 = getListLength(head1);
  int len2 = getListLength(head2);
  if (len1 == 0 || len2 == 0) return nullptr;
  ListNode *longListHead = len1 > len2 ? head1 : head2;
  ListNode *shortListHead = longListHead==head1 ? head2 : head1;
  int diff = abs(len1 - len2);
  for (int i = 0;i < diff;++i) {
    longListHead = longListHead->next;
  }
  while (longListHead != nullptr && shortListHead != nullptr && longListHead != shortListHead) {
    longListHead = longListHead->next;
    shortListHead = shortListHead->next;
  }
  ListNode*firstCom = longListHead;
  return firstCom;
}
/**********************************************************************/
//No 53 排序数组查找数字
int getFirstK(vector<int>&nums, int k, int start, int end) {
  if (start > end) return -1;
  int mid = start + (end - start) / 2;
  int midData = nums[mid];
  if (midData == k) {
    if ((mid > 0 && nums[mid - 1] != k) || mid == 0) return mid;//find first k
    else end = mid - 1;
  }
  else if (midData > k) end = mid - 1;
  else start = mid + 1;
  return getFirstK(nums, k, start, end);
}
int getLastK(vector<int>&nums, int k, int start, int end) {
  if (start > end) return -1;
  int mid = start + (end - start) / 2;
  int midData = nums[mid];
  if (midData == k) {
    if ((mid < nums.size()-1 && nums[mid + 1] != k) || mid == nums.size()-1) return mid;//find last k
    else start = mid + 1;
  }
  else if (midData > k) end = mid - 1;
  else start = mid + 1;
  return getLastK(nums, k, start, end);
}
int getNumOfK(vector<int>&nums, int k) {
  int ans = 0;
  if (nums.empty()) return ans;
  int first = getFirstK(nums, k, 0, nums.size() - 1);
  int last = getLastK(nums, k, 0, nums.size() - 1);
  if (first > -1 && last > -1)ans = last - first + 1;
  return ans;
}
/**********************************************************************/
//No 54 二叉搜索树的第k大节点
TreeNode *kthNodeCore(TreeNode*root, int &k) {
  TreeNode *target = nullptr;
  if (root->left != nullptr) target = kthNodeCore(root->left, k);
  if (target == nullptr) {
    --k;
    if (k == 0) target = root;
    if(root->right!=nullptr) target = kthNodeCore(root->right, k);
  }
  return target;
}
TreeNode *kthNode(TreeNode*root,int k) {
  if (root == nullptr || k <= 0)return nullptr;
  return kthNodeCore(root, k);
}
/**********************************************************************/
//No 55 二叉树的深度
int depthOfTree(TreeNode *root) {
  if (root == nullptr)return 0;
  int leftDep = depthOfTree(root->left);
  int rightDep = depthOfTree(root->right);
  return max(leftDep, rightDep) + 1;
}
/*********************************************************************/
//No 56 
//只出现一次的一个数

int findOnceNum(vector<int>& nums) {
  int ans = 0;
  for (auto n : nums) ans ^= n;
  return ans;
}

vector<int> findOnceNums(vector<int>& nums) {
  int xorSum = 0;
  vector<int> ans(2, 0);
  for (auto n : nums) xorSum ^= n;
  xorSum &= -xorSum;
  for (auto n : nums) {
    if (n&xorSum) ans[0] ^= n;
    else ans[1] ^= n;
  }
  return ans;
}
/***********************************************************************/
//No 57 有序数组和为s的两个数字(2 sum)
vector<int> twoSumTarget(vector<int>&nums, int s) {
  vector<int>ans;
  if (nums.size() < 2)return ans;
  int left = 0, right = nums.size() - 1;
  while (left < right) {
    if(nums[left] + nums[right] ==s){
      ans.push_back(nums[left]);
      ans.push_back(nums[right]);
      break;
    }
    else if (nums[left] + nums[right] < s) ++left;
    else --right;
  }
  return ans;
}
/***********************************************************************/
//No 58 翻转字符串，翻转句子中单词的顺序
void reverseWords(string &s) {
  reverse(s.begin(), s.end());
  int start = 0;
  for (int i = 1;i < s.size();++i) {
    if (s[i] == ' ' || s[i] == '\0') {
      reverse(s.begin() + start, s.begin() + i);
      start = i + 1;
    }
  }
  reverse(s.begin() + start, s.end());
}
/***********************************************************************/
//No 59 maximum in queue
//利用deque，队首存放当前窗口内最大值，之后是可能的最大值候选，每个数从末尾入队，删除掉所有小于它的数字
//如果队首元素不在窗口内，将其删掉
//队列存放坐标，方便范围检查
vector<int> maxInWin(vector<int> &nums, int k) {//k is the width of the window
  vector<int> maxW;
  deque<int> index;
  for (int i = 0;i < k;++i) {
    while (!index.empty() && nums[i] >= nums[index.back()]) {
      index.pop_back();
    }
    index.push_back(i);
  }
  for (int i = k;i < nums.size();++i) {
    maxW.push_back(nums[index.front()]);
    while (!index.empty() && nums[i] >= nums[index.back()]) {
      index.pop_back();
    }
    if (!index.empty() && index.front() <= (int)i - k) {
      index.pop_front();
    }
    index.push_back(i);
  }
  maxW.push_back(nums[index.front()]);
  return maxW;
}
/**********************************************************************/
//No 60 n个骰子的点数
void printProbability(int num) {
  if (num < 1)return;
  const int value = 6;
  vector<int>probs[2];
  for (int i = 0;i < 2;++i)probs[i] = vector<int>(num*value + 1,0);
  int flag = 0;
  for (int i = 1;i <= value;++i) probs[flag][i] = 1;
  for (int k = 2;k <= num;++k) {
    for (int i = 0;i < k;++i) probs[1 - flag][i] = 0;
    for (int i = k;i <= value*k;++i) {
      probs[1 - flag][i] = 0;
      for (int j = 1;j <= i&&j <= value;++j) {
        probs[1 - flag][i] += probs[flag][i-j];
      }
    }
    flag = 1 - flag;
  }
  double total = pow(value, num);
  for (int i = num;i <= value*num;++i) {
    double ratio = probs[flag][i] / total;
    cout << ratio << endl;
  }
}
/**********************************************************************/
//No 61 扑克牌中的顺子
bool isContinuous(vector<int>&nums) {
  //0---jocker
  int n = nums.size();
  if (n < 5)return false;
  sort(nums.begin(), nums.end());
  int numOfJocker = 0, numOfGap = 0;
  for (int i = 0;i < n;++i) {
    if (nums[i] == 0)++numOfJocker;
  }
  for (int i = numOfJocker+1;i < n;++i) {
    if (nums[i] == nums[i - 1]) return false;
    else {
      numOfGap += nums[i] - nums[i - 1] - 1;
    }
  }
  return numOfGap <= numOfJocker;
}
/**********************************************************************/
//No 62 圆圈最后剩下的数（约瑟夫环问题）
int lastRemaining(int n, int m) {//0-n围成环，删掉第m个重新成环，直到最后一个剩余数
  if (n < 1 || m < 1) return -1;
  int last = 0;
  for (int i = 2;i <= n;++i) {
    last = (last + m) % i;
  }
  return last;
}
/*********************************************************************/
//No 63 股票最大利润
int maxProfit(vector<int>&prices) {
  if (prices.size() <= 1) return 0;
  int buy = prices[0];
  int maxPro = prices[1] - buy;
  for (int i = 2;i < prices.size();++i) {
    buy = min(buy, prices[i-1]);
    maxPro = max(maxPro, prices[i] - buy);
  }
  return maxPro;
}

//main
/*********************************************************************/
int main() {
  //No 3
  //vector<int>nums;
  //int tmp;
  //while (cin >> tmp) nums.push_back(tmp);
  //vector<int> ans = duplicate(nums);
  //for (auto i : ans) cout << i << " ";
  //cout << endl;
  //cout << findOneDump(nums, 1, nums.size() - 1) << endl;

  //No 4
  //int target;
  //cin >> target;
  //int n, m;
  //cin >> n >> m;
  //vector<vector<int>> matrix(n, vector<int>(m));
  //for (int i = 0;i < n;++i) {
  //  for (int j = 0;j < m;++j) {
  //    cin >> matrix[i][j];
  //  }
  //}
  //cout << findNumInMatrix(matrix, target) << endl;

  //No 7
  //int n;
  //cin >> n;
  //vector<int>preorder(n);
  //vector<int>inorder(n);
  //for (int i = 0;i < n;++i) cin >> preorder[i];
  //for (int i = 0;i < n;++i) cin >> inorder[i];
  //TreeNode *root = rebuildTreeFromPreAndIn(preorder, inorder);
  //printTreeByLevel(root);

  //No 10
  //int n;
  //cin >> n;
  //for(int i =0;i<=n;++i) cout << fibonacci(i) << endl;

  //No 11
  /*int n;
  cin >> n;
  vector<int> nums(n);
  for (int i = 0;i < n;++i) cin >> nums[i];
  cout << minNumInRotateArray(nums) << endl;*/

  //No.19 正则表达式匹配
  //cout << matchRe("aaa", "aa.a") << endl;
  /***********************************************************************/

  //No.20
  /*char* str = ".e454";
  cout << isNumeric(str) << endl;*/
  //string s;
  //while (true) {
  //  cin >> s;
  //  cout << isNumeric2(s) << endl;
  //}

  //No 25
  //int m, n;
  //cin >> m >> n;
  //ListNode *head1 = inputList(m);
  //ListNode *head2 = inputList(n);
  //ListNode*ans = mergeSortedList(head1,head2);
  //outputList(ans);

  //No 29
  /*int n, m;
  cin >> n >> m;
  vector<vector<int>> matrix(n, vector<int>(m));
  for (int i = 0;i < n;++i) {
    for (int j = 0;j < m;++j) {
      cin >> matrix[i][j];
    }
  }
  printMatClkWise(matrix);*/

  //No 30
 /* int n,tmp;
  cin >> n;
  stackWithMin<int> swm;
  for (int i = 0;i < n;++i) {
    cin >> tmp;
    swm.push(tmp);
    cout << swm.getMin() << endl;
  }
  for (int i = 0;i < n;++i) {
    cout << swm.getMin() << endl;
    cout<<swm.pop()<<endl;
  }*/

  //No 31
  //int n;
  //cin >> n;
  //vector<int>n1(n), n2(n);
  //for (int i = 0;i < n;++i)cin >> n1[i];
  //for (int i = 0;i < n;++i)cin >> n2[i];
  //cout << isPopOrder(n1, n2) << endl;

  //No 32
  //TreeNode* root = genTree();
  //levelPrintTree(root);
  //levelPrintTreeRows(root);
  //levelPrintTreeZigzag(root);

  //No33
  /*int n;
  cin >> n;
  vector<int> nums(n);
  for (int i = 0;i < n;++i) cin >> nums[i];
  cout << isPostOrderOfBST(nums)<<endl;*/

  //No 34
//TreeNode* root = genTree();
//findPath(root,8);

//No 38
//string s;
//cin >> s;
//permutation(s);
//eightQueens2();
//No 39

//int n;
//cin >> n;
//vector<int> nums(n);
//for (int i = 0;i < n;++i)cin >> nums[i];
//cout << moreThanHalfNum(nums) << endl;

//No 40
//int n, k;
//cin >>k>> n;
//vector<int> nums(n);
//for (int i = 0;i < n;++i)cin >> nums[i];
//vector<int>ans = getLeastKNum2(nums, k);
//for (auto n : ans)cout << n << " ";
//cout << endl;

  //No 42
  /*int n;
  cin >> n;
  vector<int> nums(n);
  for (int i = 0;i < n;++i) cin >> nums[i];
  cout<<maxSumOfSubArray(nums)<<endl;*/

//No 43
//int n;
//cin >> n;
//cout << numOf1Between1AndN(n) << endl;

//No 44
//int n;
//cin >> n;
//cout << nthDigit(n) << endl;

//No 46
//int n;
//cin >> n;
//cout << getTranslationCount(n) << endl;

//No 47
//int n, m;
//cin >> n >> m;
//vector<vector<int>> matrix(n, vector<int>(m));
//for (int i = 0;i < n;++i) {
//  for (int j = 0;j < m;++j) {
//    cin >> matrix[i][j];
//  }
//}
//cout << maxGiftValue(matrix) << endl;

//No 48
//string s;
//cin >> s;
//cout << longestNoDup(s) << endl;

//int k;
//cin >> k;
//vector<int>nums;
//int tmp;
//while (cin >> tmp) nums.push_back(tmp);
//cout << getNumOfK(nums,k) << endl;

//int n;
//cin >> n;
//printProbability(n);


//vector<int>cards(5);
//for (int i = 0;i < 5;++i) cin >> cards[i];
//cout << isContinuous(cards) << endl;

//TreeNode *root = genTree();
//cout << kthNode(root, 3)->val << endl;

  //No 58
  //string s = "";
  //reverseWords(s);
  //cout << s << endl;

  //int tmp;
  //vector<int>nums;
  //while (cin >> tmp) nums.push_back(tmp);
  //vector<int>ans = findOnceNums(nums);
  //cout << ans[0]<<", "<<ans[1]<< endl;
  //string t;
  //cin >> t;
  //cout << firstNotRepeat(t) << endl;
  //int k, tmp;
  //cin >> k;
  //vector<int>nums;
  //while (cin >> tmp) nums.push_back(tmp);
  //vector<int>ans = maxInWin(nums, k);
  //for (auto i : ans) cout << i << " ";
  //cout << endl;
  return 0;
}
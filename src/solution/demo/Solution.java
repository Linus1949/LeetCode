package solution.demo;
import javafx.util.Pair;

import java.util.*;


public class Solution {
    /**
     * Basic Struct
     */
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }
    public class ListNode{
        int val;
        ListNode next;

        ListNode(int val){
            this.val = val;
        }
    }
    public class TreeLinkNode{
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode parent = null;

        TreeLinkNode(int val){
            this.val = val;
        }
    }

    // leetCode 910
    class StockSpanner {
        Stack<Integer> prices, days;

        public StockSpanner() {
            prices = new Stack<Integer>();
            days = new Stack<Integer>();
        }

        public int next(int price) {
            int day = 1;
            while (!prices.isEmpty() && prices.peek() <= price) {
                prices.pop();
                day += days.pop();
            }
            prices.push(price);
            days.push(day);
            return day;
        }
    }


    //LeetCode 1137
    public int tribonacci(int n) {
        int[] dp = new int[38];
        dp[0] = 0;
        dp[1] = 1;
        dp[2] = 1;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3];
        }
        return dp[n];
    }

    /**
     * LeetCode 29
     */
    public int devide(int dividend, int divisor) {
        boolean sign = (dividend > 0) ^ (divisor > 0);
        int result = 0;
        //避免正数边界统一用负数计算
        if (dividend > 0) {
            dividend = -dividend;
        }
        if (divisor > 0) {
            divisor = -divisor;
        }
        //但两个数字都为负数时，dividend大于divisor
        while (dividend <= divisor) {
            int tempResult = -1;
            int tempDivisor = divisor;
            //尝试将tempDivisor翻倍
            while (dividend <= (tempDivisor << 1)) {
                //当tempDivisor小于等于边界两倍时必须退出，不然下一次循环就会触及边界
                if (tempDivisor <= (Integer.MIN_VALUE >> 1)) {
                    break;
                }
                //每次循环将
                tempResult = tempResult << 1;
                tempDivisor = tempDivisor << 1;
            }
            dividend = dividend - tempDivisor;
            result += tempResult;
        }
        if (!sign) {
            if (result <= Integer.MIN_VALUE) {
                return Integer.MAX_VALUE;
            }
            result = -result;
        }
        return result;
    }

    /**
     * 本题的要求时在不使用除法，乘法和mod的基础上实现除法，虽然我们可以使用减法直接解决，可是效率太低，因此
     * 我们可以尝试使用>>位移来使除数翻倍，用dividened除以2尝试不断地减小n来满足dividened/2^n >= divisor
     * 表示我们找到了一个足够大的数，，这个数*divisor是不大于dividend的，所以我们就可以减去2^n个divisor
     * 这其中得处理一些特殊的数，比如divisor是不能为0的，Integer.MIN_VALUE和Integer.MAX_VALUE
     */
    public int devideV2(int dividend, int divisor) {
        if (dividend == 0) {
            return 0;
        }
        if (dividend == Integer.MIN_VALUE && divisor == -1) {
            return Integer.MAX_VALUE;
        }
        boolean negative;
        negative = (dividend ^ divisor) < 0;
        long t = Math.abs((long) dividend);
        long d = Math.abs((long) divisor);
        int result = 0;
        for (int i = 31; i >= 0; i--) {
            if ((t >> i) >= d) {
                result += 1 << i;
                t -= d << i;
            }
            // 添加一个判断
            if (t < divisor) {
                break;
            }
        }
        return negative ? -result : result;
    }

    /**
     * LeetCode:329
     * Level:Hard
     * DFS + 记忆化
     */
    public int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    public int rows, cols;

    public int dfs(int[][] matrix, int row, int col, int[][] memo) {
        //记忆化
        if (memo[row][col] != 0) {
            return memo[row][col];
        }
        //路过matrix[row][col]本身
        ++memo[row][col];
        for (int[] dir : dirs) {
            int newRows = row + dir[0];
            int newCols = col + dir[1];
            //判断边界并且下一个节点必须要递曾
            if (newRows >= 0 && newRows < rows && newCols >= 0 && newCols < cols && matrix[newRows][newCols] > matrix[row][col]) {
                memo[row][col] = Math.max(memo[row][col], dfs(matrix, newRows, newCols, memo) + 1);
            }
        }
        return memo[row][col];
    }

    public int longestIncreasingPath(int[][] matrix) {
        //base case
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return 0;
        }
        rows = matrix.length;
        cols = matrix[0].length;
        //记忆化
        int[][] memo = new int[rows][cols];
        int res = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res = Math.max(res, dfs(matrix, i, j, memo));
            }
        }
        return res;
    }

    /**
     * LeetCode:392
     * Level:Easy
     */
    public boolean isSubsequence(String s, String t) {
        int sLen = s.length();
        int tLen = t.length();
        if (sLen > tLen) {
            return false;
        }
        if (sLen == 0 || tLen == 0) {
            return true;
        }
        int sIndex = 0;
        for (int i = 0; i < tLen; i++) {
            if (t.charAt(i) == s.charAt(sIndex)) {
                if (++sIndex == sLen) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * LeetCode 406
     * Level: Medium
     * 题解：因为身高矮的人相对身高高的人是不影响k的，所以我们优先身高排序，如果身高相同我们以K升序排列，
     * 相应矮的人插入对应K的位置是保证结果中的K不会被破坏，因为结果中高的人已经站好了，矮的人无法影响他们，但因为矮的人
     * 要占据一个位置所以高的人要先后移
     */
    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] o1, int[] o2) {
                return o1[0] == o2[0] ? o1[1] - o2[1] : o2[0] - o1[0];
            }
        });
        List<int[]> list = new LinkedList<>();
        for (int[] p : people) {
            list.add(p[1], p);
        }
        return list.toArray(new int[people.length][2]);
    }

    /**
     * LeetCode 104
     * Level: Easy
     */
    public int maxDepth(TreeNode root) {
        return TreeDepth(root);
    }

    //方法一：递归
    private int TreeDepth(TreeNode node) {
        if (node == null) {
            return 0;
        }
        return Math.max(TreeDepth(node.left), TreeDepth(node.right)) + 1;
    }

    //方法二：层序遍历
    public int maxDepth2(TreeNode root) {
        if (root == null) {
            return 0;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int maxDepth = 0;
        while (!queue.isEmpty()) {
            maxDepth++;
            int levelSize = queue.size();
            for (int i = 0; i < levelSize; i++) {
                TreeNode temp = queue.pollFirst();
                if (temp.left != null) {
                    queue.add(temp.left);
                }
                if (temp.right != null) {
                    queue.add(temp.right);
                }
            }
        }
        return maxDepth;
    }

    /**
     * LeetCode 94
     * Level: Medium
     * 中序遍历，使用Stack
     */
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        if (root == null) {
            return res;
        }
        Deque<TreeNode> stack = new ArrayDeque<>();
        TreeNode temp = root;
        while (!stack.isEmpty() || temp != null) {
            while (temp != null) {
                stack.addFirst(temp);
                temp = temp.left;
            }
            temp = stack.removeFirst();
            res.add(temp.val);
            temp = temp.right;
        }
        return res;
    }

    /**
     * LeetCode LCP 13 寻宝
     * Level: Hard
     */
//    public class SearchItem {
//        public Integer row;
//        public Integer col;
//        public Integer cost;
//
//        SearchItem(int row, int col, int cost) {
//            this.row = row;
//            this.col = col;
//            this.cost = cost;
//        }
//    }
//
//    public String[][] grip;
//    public HashMap<String, String> map;
//
//    public int minimaSteps(String[] maze) {
//        //初始化map方便查询
//        map = new HashMap<String, String>();
//        map.put("start", "S");
//        map.put("wall", "#");
//        map.put("stone", "O");
//        map.put("pass", ".");
//        map.put("target", "T");
//        map.put("trick", "M");
//        //将原始数组转换成二维数组方便dfs
//        grip = new String[maze.length][maze[0].length()];
//        for (int i = 0; i < maze.length; i++) {
//            for (int j = 0; j < maze[0].length(); j++) {
//                grip[i][j] = Character.toString(maze[i].charAt(j));
//            }
//        }
//        //
//        return 0;
//    }
//
//    private SearchItem search(String[][] grip, int row, int col, int cost, String type) {
//        if (grip[row][col].equals(map.get(type))) {
//            return new SearchItem(row, col, cost);
//        }
//        for (int[] dir : dirs) {
//            int newRow = row + dir[0];
//            int newCol = col + dir[1];
//            //判断边界
//            if (newRow >= 0 && newRow < grip.length && newCol >= 0 && newCol < grip[0].length) {
//                return search(grip, newRow, newCol, cost + 1, type);
//            }
//        }
//        return null;
//    }

    /**
     * LeetCode 617, 合并二叉树
     * Level: Easy
     * 层序遍历
     */
    public TreeNode mergeTrees(TreeNode t1, TreeNode t2){
        //base case
        if(t1==null || t2==null){
            return t1==null? t2: t1;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(t1);
        queue.add(t2);
        while (!queue.isEmpty()){
            TreeNode tempOne = queue.remove();
            TreeNode tempTwo = queue.remove();
            tempOne.val += tempTwo.val;
            //如果left都不为null
            if(tempOne.left!=null && tempTwo.left!=null) {
                queue.add(tempOne.left);
                queue.add(tempTwo.left);
            }else if(tempOne.left==null){
                tempOne.left = tempTwo.left;
            }
            //如果right都不为null
            if(tempOne.right!=null && tempTwo.right!=null){
                queue.add(tempOne.right);
                queue.add(tempTwo.right);
            }else if(tempOne.right==null){
                tempOne.right = tempTwo.right;
            }
        }
        return t1;
    }
    /**
     * LeetCode 343，整数拆分
     * Level: Medium
     * 假设将数字x拆分成a份，n = ax, 乘积便是 x^a = x^(n/x) = (x^(1/x))^n
     * 因此乘积最大，就是让x^(1/x)最大，对它求导
     * In(x) = 1/x * In(x)
     * x' = (1-ln(x) / x^2) * (x^(1/x))
     * 令x' = 0, 1 - ln(x) = 0, e = 2.7
     * x = 3 最大, x = 2次子, x = 1最差
     */
    public int integerBreak(int n) {
        //当n<=3时，丢弃一个因子，返还n-1
        if (n <= 3) {
            return n - 1;
        }
        //当n>3时，得到商a和余数b, n = 3a + b
        int a = n / 3, b = n % 3;
        //当b为0时，直接返还3^a即可
        if (b == 0) {
            return (int) Math.pow(3, a);
        }
        //当b为1时，要将最后一个1+3 转换成 2+2, n = 3^(a-1)*4
        if (b == 1) {
            return (int) Math.pow(3, a - 1) * 4;
        }
        //当b为2时，n = 3^a * 2, 因为2为第二选择, 不需要拆分成1+1
        return (int) Math.pow(3, a) * 2;
    }
    /**
     * LeetCode 704, 二分查找
     * Level: Easy
     */
    public int search(int[] nums, int target){
        int left = 0;
        int right = nums.length-1;
        while(left<=right){
            int mid = (right-left)/2 + left;
            if(nums[mid]==target) {
                return mid;
            }else if(nums[mid]<target){
                left = mid+1;
            }else{
                right = mid-1;
            }
        }
        return -1;
    }
    /**
     * LeetCode -, 魔术索引
     * Level: Easy
     */
    public int findMagicIndex(int[] nums){
        if(nums==null || nums.length==0){
            return -1;
        }
        for(int i=0;i< nums.length;i++){
            if(nums[i]==i){
                return i;
            }
        }
        return -1;
    }
    /**
     * 对称二叉树，与此树的镜像相同就是对称的，也就是说从根节点开始如果左右子树不等就是
     */
    public boolean isSymmetrical(TreeNode pRoot){
       //base case: empty tree is symmetrical
        if (pRoot==null){
            return true;
        }
        //we need two queues to traverse left and right
        LinkedList<TreeNode> leftTree = new LinkedList<>();
        LinkedList<TreeNode> rightTree = new LinkedList<>();
        leftTree.add(pRoot);
        rightTree.add(pRoot);
        while(!leftTree.isEmpty() && !rightTree.isEmpty()){
            TreeNode tempOne = leftTree.poll();
            TreeNode tempTwo = rightTree.poll();
            //three cases
            if(tempOne==null && tempTwo==null){
                continue;
            }
            if (tempOne==null || tempTwo==null){
                return false;
            }
            if(tempOne.val != tempTwo.val){
                return false;
            }
            //traverse
            leftTree.add(tempOne.left);
            rightTree.add(tempTwo.right);
            //opposite direction
            leftTree.add(tempOne.right);
            rightTree.add(tempTwo.left);
        }
        return true;
    }
    /**
     * 二叉树的下一个节点，给定一个节点，以中序遍历的顺序返回下一个节点，注：这棵树的节点除了有左右子节点还有指向父节点的指针
     *                  A
     *          B               C
     *     D        E       F       G
     *          H       I
     * 中序顺序: D, B, H, E, I, A, F, C, G
     */
    public TreeLinkNode GetNext(TreeLinkNode pNode){
        //case one:当前节点有右子树，下一个系欸但就是右子树的最左节点；例如B，下一个节点为H
        if(pNode.right!=null){
            TreeLinkNode pRight = pNode.right;
            while(pRight.left!=null){
                pRight = pRight.left;
            }
            return pRight;
        }
        //case two:当前节点没有右子树，且该节点在父节点的左子树，则下一个节点就是该父节点；例如H，下一个节点为E
        if(pNode.parent!=null && pNode.parent.left==pNode){
            return pNode.parent;
        }
        //case three:当前节点没有右子树，且该节点在其父节点的右子树，则我们一直沿着父节点回溯，直到找到某节点，如果其父节点在这个节点的左子树上，那么下一个节点就是这个节点；例如I，下一个节点为A
        //如果不是则为空，因为当前节点是其父节点的最右边且父节点是某节点的最右边，因此再没有其他节点会在该节点后面；例如G，下一个节点为null
        if(pNode.parent!=null){
            TreeLinkNode pParent = pNode.parent;
            while(pParent.parent!=null && pParent.parent.right==pParent){
                pParent = pParent.parent;
            }
            return pParent.parent;
        }
        return null;
    }
    /**
     * 猿辅导2020第一次笔试，参考：LeetCode 253, 求最少需要多少个会议室
     * 首先对每个课程的起始时间进行排序，使用最小堆维护当前课程的结束时间
     * 当新的课程出现。我们需要判断新的课程是否与前面的课程时间重叠,记录最小堆的历史最大size即可
     */
    public int minK(List<int[]> list){
        //排序
        list.sort((o1, o2) -> o1[0]-o2[0]);
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        int res = 0;
        for (int[] ints : list) {
            if(!minHeap.isEmpty() && minHeap.peek()<=ints[0]){
                minHeap.poll();
            }
            minHeap.add(ints[1]);
            res = Math.max(res, minHeap.size());
        }
        return res;
    }
    /**
     * 多多鸡有N个魔术盒子（编号1～N），其中编号为i的盒子里有i个球。
     * 多多鸡让皮皮虾每次选择一个数字X（1 <= X <= N），多多鸡就会把球数量大于等于X个的盒子里的球减少X个。
     * 通过观察，皮皮虾已经掌握了其中的奥秘，并且发现只要通过一定的操作顺序，可以用最少的次数将所有盒子里的球变没。
     * 那么请问聪明的你，是否已经知道了应该如何操作呢？
     */
    public int minTimes(int n){
        //当仅有1个盒子时，只需操作1次
        if(n==1){
            return 1;
        }
        //当有2个盒子时，需要两次操作才能清空，因为第2个盒子会比第一个多1个球
        if(n==2){
            return 2;
        }
        //我们每次都从队列中挑选出中止，这样前后的盒子里的球就会相同，所以相当于将所有盒子2分，所以每次二分就是一次操作
        return 1+minTimes(n/2);
    }
    /**
     * 多多鸡打算造一本自己的电子字典，里面的所有单词都只由a和b组成。
     * 每个单词的组成里a的数量不能超过N个且b的数量不能超过M个。
     * 多多鸡的幸运数字是K，它打算把所有满足条件的单词里的字典序第K小的单词找出来，作为字典的封面。
     */
    public int topK(int n, int m, int k){
        return 0;
    }
    /**
     * LeetCode 114, 二叉树展开为链表
     * Level: medium
     */
    public void flatten(TreeNode root){
        if(root==null){
            return;
        }
        while (root!=null){
            if(root.left==null){
                root = root.right;
            }else{
                //找到左子树的最右节点
                TreeNode pre = root.left;
                while (pre.right!=null){
                    pre = pre.right;
                }
                pre.right = root.right;
                root.right = root.left;
                root.left = null;
                root = root.right;
            }
        }
    }
    /**
     * 删除链表中重复的节点，所有重复的节点都不保留
     */
    public ListNode deleteDuplication(ListNode pHead)
    {
        if(pHead==null || pHead.next==null){
            return pHead;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = pHead;
        ListNode pre = dummy;
        ListNode last = dummy.next;
        while(last!=null){
            //不是tie的重复节点
            if(last.next!=null && last.val == last.next.val){
                //找到最后一个相同节点
                while(last.next!=null && last.val == last.next.val){
                    last = last.next;
                }
                pre.next = last.next;
                last = last.next;
            }else{
                pre = pre.next;
                last = last.next;
            }
        }
        return dummy.next;
    }
    /**
     * LeetCode 164, 最大间距
     * Level: Hard
     */
    //方法1：比较排序，复杂度难以突破O(nlogn)
    public int maximumGapOne(int[] nums){
        if(nums.length<2){
            return 0;
        }
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for(int num: nums){
            minHeap.add(num);
        }
        int max = Integer.MIN_VALUE;
        int num = minHeap.poll();
        while (!minHeap.isEmpty()){
            max = Math.max(max, Math.abs(num-minHeap.peek()));
            num = minHeap.poll();
        }
        return max;
    }
    //方法2桶排序，复杂度O(n)，在桶排序中，每个桶的区间长度一般都是一样的，比如说给定数组 [1,5,7,10]，
    // 这里如果我们分 10 个桶，那么每个桶的区间长度就是 1，等同于每个桶其实就对应一个数，如果这里我们分 1 个桶，
    // 那么这个桶的区间范围就是 1 ~ 10，当然这里我给的两个例子都是极端的例子，在实际应用上我们应该结合实际情况合理分配桶。
    //构建桶结构
    private class Bucket{
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
    }
    public int maximumGapTwo(int[] nums){
        //base case
        if(nums==null || nums.length<2){
            return 0;
        }
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for(int num:nums){
            min = Math.min(min, num);
            max = Math.max(max, num);
        }
        //如何分配桶的长度与个数是关键
        //在n个数下，形成的区间是n-1，例如[2,4,6,8]
        //这里有4个数，但是只有3个区间，[2,4], [4,6], [6,8]
        //因此，桶长度 = 区间长度/区间个数 = (max - min) / (nums.length - 1)
        int bucketSize = Math.max(1, (max-min)/(nums.length-1));

        // 上面得到了桶的长度，我们就可以以此来确定桶的个数
        // 桶个数 = 区间长度 / 桶长度
        // 这里考虑到实现的方便，多加了一个桶，为什么？
        // 还是举上面的例子，[2,4,6,8], 桶的长度 = (8 - 2) / (4 - 1) = 2
        //                           桶的个数 = (8 - 2) / 2 = 3
        // 已知一个元素，需要定位到桶的时候，一般是 (当前元素 - 最小值) / 桶长度
        // 这里其实利用了整数除不尽向下取整的性质
        // 但是上面的例子，如果当前元素是 8 的话 (8 - 2) / 2 = 3，对应到 3 号桶
        //              如果当前元素是 2 的话 (2 - 2) / 2 = 0，对应到 0 号桶
        // 你会发现我们有 0,1,2,3 号桶，实际用到的桶是 4 个，而不是 3 个
        // 透过例子应该很好理解，但是如果要说根本原因，其实是开闭区间的问题
        // 这里其实 0,1,2 号桶对应的区间是 [2,4),[4,6),[6,8)
        // 那 8 怎么办？多加一个桶呗，3 号桶对应区间 [8,10)
        Bucket[] buckets = new Bucket[(max - min) / bucketSize + 1];

        //对桶进行填充，插入桶的时候，我们就已经在排序了，一部分数字进了相同的桶，直接通过max,min对比
        for (int num : nums) {
            int temp = (num - min) / bucketSize;

            if (buckets[temp] == null) {
                buckets[temp] = new Bucket();
            }
            buckets[temp].min = Math.min(buckets[temp].min, num);
            buckets[temp].max = Math.max(buckets[temp].max, num);
        }
        //另一部分数字不在相同的桶，我们通过比较相邻的桶即可
        int previousMax = Integer.MAX_VALUE; int maxGap = Integer.MIN_VALUE;
        for (Bucket bucket : buckets) {
            if (bucket != null && previousMax != Integer.MAX_VALUE) {
                maxGap = Math.max(maxGap, bucket.min - previousMax);
            }
            if (bucket != null) {
                previousMax = bucket.max;
                maxGap = Math.max(maxGap, bucket.max - bucket.min);
            }
        }
        return maxGap;
    }
    /**
     * LeetCode 415, 字符串相加
     * Level: Easy
     */
     public String addString(String num1, String num2){
         if(num1==null && num2==null){
             return new String("0");
         }
         else if(num1==null){
             return num2;
         }else if(num2==null){
             return num1;
         }else{
             int num1Ptr = num1.length()-1;
             int num2Ptr = num2.length()-1;
             int up = 0;
             StringBuffer sb = new StringBuffer();
             while(num1Ptr>=0 || num2Ptr>=0 || up!=0){
                 int x = num1Ptr>=0? num1.charAt(num1Ptr) - 48: 0;
                 int y = num2Ptr>=0? num2.charAt(num2Ptr) - 48: 0;
                 int res = x+y+up;
                 //如果超过10，取余
                 sb.append(res%10);
                 up = res/10;
                 num1Ptr--;
                 num2Ptr--;
             }
             sb.reverse();
             return sb.toString();
         }
     }
     /**
      * LeetCode 124, 二叉树中的最大路径和
      * Level: Hard
     */
     public int maxPathSum = Integer.MIN_VALUE;
     public int maxPathSum(TreeNode root){
         maxGrain(root);
         return maxPathSum;
     }
     public int maxGrain(TreeNode node){
         //base case
         if(node==null){
             return 0;
         }
         //递归取得左右子树的最大路径和，如果出现负数节点，直接ignore
         int leftGain = Math.max(0, maxGrain(node.left));
         int rightGain = Math.max(0, maxGrain(node.right));
         //update
         maxPathSum = Math.max(maxPathSum, node.val+leftGain+rightGain);
         //作为路径只能返回左右子树的一个
         return node.val + Math.max(leftGain, rightGain);
     }
    /**
     * LeetCode 53, 最大子序和
     * Level: Easy
     * dp[i]表示以i结尾的子串的最大值
     * 例如第一个数字结尾的连续序列，[-2], 最大值：-2
     * 第二个数字结尾的连续序列，[-2,1], [1], 最大值：1
     * 第三个数字结尾的连续序列，[-2,1,3], [1,3], [3], 最大值:3
     * 每次向后增加一位，就在上一级的每个子序列里都加入新的数字，并且新增一个当前数字的子序列。如果上一级是负数，现在的数无论正负，都是数字自己本身更大，而如果上一级是正的，那么一定是上一级最优子序列+现在的数字最优
     * if(dp[i-1])>0? dp[i] = dp[i-1] + nums[i]： dp[i] = nums[i]
     */
    public int maxSubArray(int[] nums){
        int len = nums.length;
        if (len==0){
            return 0;
        }
        int[] dp = new int[len];
        dp[0] = nums[0];
        for(int i=1;i<len;i++){
            if(dp[i-1]>0){
                dp[i] = dp[i-1] + nums[i];
            }else{
                dp[i] = nums[i];
            }
        }
        //遍历寻找最大值
        int maxSum = Integer.MIN_VALUE;
        for(int num:dp){
            maxSum = Math.max(maxSum, num);
        }
        return maxSum;
    }
    /**
     * LeetCode 141, 环形链表
     * Level: Easy
     */
    public boolean hasCycle(ListNode head){
        //base case
        if (head==null){
            return false;
        }
        ListNode fast = head.next;
        ListNode slow = head;
        while (slow!=fast){
            if (fast==null || fast.next==null){
                return false;
            }
            slow = slow.next;
            fast = fast.next.next;
        }
        return true;
    }
    /**
     * leetCode 337, 打家劫舍III
     * Level: Medium
     * 思路：首先每个节点都有偷/不偷两个状态，并且父节点偷，两个子节点就不能偷，如果父节点不偷，那么两个子节点可以选择偷或者不偷
     * 0代表该节点不偷，1代表偷
     * res[0] = Math.max(Math.max(left[0],left[1]), Math.max(right[0),right[1))
     * res[1] = node.val + left[0] + right[0]
     */
    public int rob(TreeNode root){
        int[] ans = robTrack(root);
        return Math.max(ans[0], ans[1]);
    }
    public int[] robTrack(TreeNode node){
        if(node==null){
            return new int[2];
        }
        //每个节点保存偷/不偷两种状态
        int[] res = new int[2];
        //递归
        int[] left = robTrack(node.left);
        int[] right = robTrack(node.right);
        //当父节点不偷, 左右子节点可以选择偷/不偷
        res[0] = Math.max(left[0],left[1]) + Math.max(right[0],right[1]);
        //当父节点偷，那么左右子节点都不能偷
        res[1] = node.val + left[0] + right[0];
        return res;
    }
    /**
     * LeetCode 300, 最长上升子序列
     * Level: Medium
     */
    //方法1：dp[i]表示前i给数字的最长上升序列长度，复杂度: O(n^2)
    public int lengthOfLISOne(int[] nums){
        int len = nums.length;
        //dp[i]: 前i个数字的最长升序子序列长度、
        int[] dp = new int[len];
        //因为数字本身也是长度的一部分
        int maxLen = 0;
        Arrays.fill(dp,1);
        for(int i=0;i<len;i++){
            //从后向前遍历
            for(int j=0;j<i;j++){
                if(nums[j]<nums[i]){
                    dp[i] = Math.max(dp[i], dp[j]+1);
                }
            }
            maxLen = Math.max(maxLen, dp[i]);
        }
        return  maxLen;
    }
    //方法2, 贪婪+二分查找，
    //https://leetcode-cn.com/problems/longest-increasing-subsequence/solution/dong-tai-gui-hua-er-fen-cha-zhao-tan-xin-suan-fa-p/
    public int lengthOfLIS(int[] nums) {
        int len = nums.length;
        if (len <= 1) {
            return len;
        }
        // tail 数组的定义：长度为 i + 1 的上升子序列的末尾最小是几
        int[] tail = new int[len];
        // 遍历第 1 个数，直接放在有序数组 tail 的开头
        tail[0] = nums[0];
        // end 表示有序数组 tail 的最后一个已经赋值元素的索引
        int end = 0;

        for (int i = 1; i < len; i++) {
            // 【逻辑 1】比 tail 数组实际有效的末尾的那个元素还大
            if (nums[i] > tail[end]) {
                // 直接添加在那个元素的后面，所以 end 先加 1
                end++;
                tail[end] = nums[i];
            } else {
                // 使用二分查找法，在有序数组 tail 中
                // 找到第 1 个大于等于 nums[i] 的元素，尝试让那个元素更小
                int left = 0;
                int right = end;
                while (left < right) {
                    // 选左中位数不是偶然，而是有原因的，原因请见 LeetCode 第 35 题题解
                    // int mid = left + (right - left) / 2;
                    int mid = left + ((right - left) >>> 1);
                    if (tail[mid] < nums[i]) {
                        // 中位数肯定不是要找的数，把它写在分支的前面
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                // 走到这里是因为 【逻辑 1】 的反面，因此一定能找到第 1 个大于等于 nums[i] 的元素
                // 因此，无需再单独判断
                tail[left] = nums[i];
            }
            // 调试方法
            // printArray(nums[i], tail);
        }
        // 此时 end 是有序数组 tail 最后一个元素的索引
        // 题目要求返回的是长度，因此 +1 后返回
        end++;
        return end;
    }
    /**
     * 牛客网 字符串统计
     * 输入一个字符串，以出现次数由高到低的输出，如果出现次数相同按照ASCII的顺序打印
     * 因为ASCII一共只有256个字符，因此可以通过 int[256] 来捕捉位置
     */
    public void printASICC(String str){
        //pos捕捉字符，value记录出现次数
        int[] count = new int[256];
        int max = 0;
        for(int i=0;i<str.length();i++){
            count[str.charAt(i)]++;
            max = Math.max(max, count[str.charAt(i)]);
        }
        //循环由高到低打印，最后按照ASCII顺序打印
        while (max!=0){
            for(int i=0;i<256;i++){
                if(count[i]==max){
                    System.out.print((char)(i));
                }
            }
            max--;
        }
        System.out.println();
    }
    /**
     * LeetCode 110，平衡二叉树
     * Level: Easy
     */
    public boolean isBalanced(TreeNode root){
        //base case
        if(root==null || (root.left==null && root.right==null)){
            return true;
        }
        int leftDepth = Depth(root.left);
        int rightDepth = Depth(root.right);
        //如果完全平衡，直接返回
        if(Math.abs(leftDepth - rightDepth) < 1){
            return true;
        }
        //如果不完全平衡，需要确定每一个子树都是平衡的才行
        return isBalanced(root.left) && isBalanced(root.right);
    }
    //获取子树最大深度
    public int Depth(TreeNode node){
        if(node==null){
            return 0;
        }
        return Math.max(Depth(node.left), Depth(node.right))+1;
    }
    /**
     * LeetCode 130，被围绕的区域
     * Level：Medium
     * BFS
     */
    public void solve(char[][] board){
        if (board==null || board.length==0){
            return;
        }
        int rows = board.length;
        int cols = board[0].length;
        //通过bfs将联通的O全部先转换为T
        for (int col=0;col<cols;col++){
            bfs(board, 0, col);
            bfs(board,rows-1,col);
        }
        for(int row=0;row<rows;row++){
            bfs(board,row,0);
            bfs(board,row,cols-1);
        }
        //将不满足条件的O全部转换成X
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                if(board[i][j]=='O'){
                    board[i][j] = 'X';
                }
            }
        }
        //再把被标记为T的部分转换回O
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                if(board[i][j]=='T'){
                    board[i][j] = 'O';
                }
            }
        }
    }
    public void bfs(char[][] board, int row, int col){
        //判断边界，满足条件就可以bfs继续搜索
        if(row>=0 && row<board.length && col>=0 && col< board[0].length && board[row][col]=='O'){
            board[row][col] = 'T';
            bfs(board, row+1, col);
            bfs(board, row-1, col);
            bfs(board, row, col+1);
            bfs(board, row, col-1);
        }
    }
    /**
     * LeetCode 336, 回文对
     * Level: Hard
     * 假设有两个字符串，判断s1+s2是否为回文字符串，有以下三种case
     * 1. len1 == len2, 判断s1是否为s2翻转即可
     * 2. len1 > len2, 将s1分成t1和t2，判断t1是s2的翻转，那么t2是一个回文字符串
     * 3. len1 < len2, 将s2分成t1和t2, 判断t2是s1的翻转，那么t1是一个回文字符串
     * 因此我们需要枚举每一个字符串k, 注意空字符串也是回文字符串，我们使用字典树来
     * 辅助进行翻转判断
     */
    //我们可以使用哈希表存储所有字符串的翻转串。在进行查询时，我们判断带查询串的子串是否在哈希表中出现，就等价于判断了其翻转是否存在。
    List<String> wordsRev = new ArrayList<>();
    HashMap<String, Integer> indicate = new HashMap<>();
    public List<List<Integer>> palindromePairs(String[] words){
        int len = words.length;
        //将单词翻转储存，方便回文配对
        for(String word:words){
            wordsRev.add(new StringBuffer(word).reverse().toString());
        }
        //将翻转好的单词与他们在原始数组中的位置绑定
        for(int i=0;i<len;i++){
            indicate.put(wordsRev.get(i),i);
        }
        List<List<Integer>> res = new ArrayList<>();
        //遍历每个单词
        for(int i=0;i<len;i++){
            String word = words[i];
            int wordLen = word.length();
            if(wordLen==0){
                continue;
            }
            //将每个单词都遍历所有的substring,当单词不一样时，需要切分成t1,t2两部分
            //分别尝试是否可以单独与其他单词进行回文匹配
            for(int j=0;j<=wordLen;j++){
                //将单词切成j到wordLen-1的t2
                if(isPalindrome(word,j,wordLen-1)){
                    //获取与t2能组合的单词id
                    int leftId = findWord(word,0,j-1);
                    //排除没找到和找到的是自己本身的情况
                    if(leftId!=-1 && leftId!=i){
                        res.add(Arrays.asList(i,leftId));
                    }
                }
                //将单词切成从0到j的t1
                if(j!=0 && isPalindrome(word,0,j-1)){
                    //获取与t1能组合的单词
                    int rightId = findWord(word,j,wordLen-1);
                    //排除没找找到和找到的是自己本身的情况
                    if(rightId!=-1 && rightId!=i){
                        res.add(Arrays.asList(rightId,i));
                    }
                }
            }
        }
        return res;
    }
    //判断是否是一串回文字符串
    public boolean isPalindrome(String word, int left, int right){
        int len = right - left + 1;
        for(int i=0;i<len;i++){
            if(word.charAt(left+i)!=word.charAt(right-i)){
                return false;
            }
        }
        return true;
    }
    //寻找是否有与之匹配的翻转字符串, 如果有返回相应的位置，没有返回-1
    public int findWord(String word, int left, int right){
        return indicate.getOrDefault(word.substring(left,right+1),-1);
    }
    /**
     * LeetCode 165, 比较版本号
     * Level：Medium
     * 暴力扩容，然后遍历通过regex修饰通过
     */
    public int compareVersionTwo(String version1, String version2){
        String[] version1Block = version1.split("\\.");
        String[] version2Block = version2.split("\\.");
        int blockOneLen = version1Block.length;
        int blockTwoLen = version2Block.length;
        //将长度对齐
        if(blockOneLen!=blockTwoLen){
            int minLen = Math.min(blockOneLen,blockTwoLen);
            int disLen = Math.abs(blockOneLen-blockTwoLen);
            if(minLen==blockOneLen){
                //扩容
                version1Block = Arrays.copyOf(version1Block,blockOneLen+disLen);
                for (int i=blockOneLen;i<blockOneLen+disLen;i++){
                    version1Block[i] = new String("0");
                }
            }else{
                //扩容
                version2Block = Arrays.copyOf(version2Block,blockTwoLen+disLen);
                for (int i=blockTwoLen;i<blockTwoLen+disLen;i++){
                    version2Block[i] = new String("0");
                }
            }
        }
        //开始对比
        for(int i=0;i<version1Block.length;i++){
            int v1 = Integer.parseInt(version1Block[i]);
            int v2 = Integer.parseInt(version2Block[i]);
            if(!"0".equals(version1Block[i])){
                if("".equals(version1Block[i].replaceAll("^(0+)", ""))){
                    v1 = 0;
                }else{
                    v1  = Integer.parseInt(version1Block[i].replaceAll("^(0+)",""));
                }
            }
            if(!"0".equals(version2Block[i])){
                if("".equals(version2Block[i].replaceAll("^(0+)", ""))){
                    v2 = 0;
                }else{
                    v2 = Integer.parseInt(version2Block[i].replaceAll("^(0+)",""));
                }
            }
            if(v1>v2){
                return 1;
            }
            if(v1<v2){
                return -1;
            }
        }
        return 0;
    }
    /**
     * LeetCode 165, 比较版本号
     * Level：Medium
     * 维护两个指针分别指向每个字符串中的每个chunk
     */
    public int compareVersion(String version1, String version2){
        //双指针，负责遍历时指向每一个字符串的子串
        int p1 = 0, p2 = 0;
        int len1 = version1.length(), len2 = version2.length();

        //compare
        int i1, i2;
        Pair<Integer, Integer> pair;
        while(p1<len1 || p2<len2){
            pair = getNextChunk(version1, len1, p1);
            i1 = pair.getKey();
            p1 = pair.getValue();

            pair = getNextChunk(version2,len2,p2);
            i2 = pair.getKey();
            p2 = pair.getValue();
            if(i1!=i2){
                return i1 > i2? 1:-1;
            }
        }
        return 0;
    }
    public Pair<Integer, Integer> getNextChunk(String version, int n, int p){
        //base case
        if(p>n-1){
            return new Pair(0,p);
        }
        //find the end of chunk
        int i, pEnd = p;
        while (pEnd<n && version.charAt(pEnd)!='.'){
            ++pEnd;
        }
        //取回chunk
        if(pEnd!=n-1){
            i = Integer.parseInt(version.substring(p,pEnd));
        }else{
            i = Integer.parseInt(version.substring(p,n));
        }
        //获取下一个chunk的起始点
        p = pEnd+1;

        return new Pair(i, p);
    }
    /**
     * LeetCode 100, 相同的树
     * Level：Easy
     */
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        LinkedList<TreeNode> stackP = new LinkedList<>();
        LinkedList<TreeNode> stackQ = new LinkedList<>();
        stackP.addFirst(p);
        stackQ.addFirst(q);
        while (!stackP.isEmpty() && !stackQ.isEmpty()) {
            TreeNode tempP = stackP.removeFirst();
            TreeNode tempQ = stackQ.removeFirst();

            if (tempP.val != tempQ.val) {
                return false;
            }

            if (tempP.left != null && tempQ.left != null) {
                stackP.addFirst(tempP.left);
                stackQ.addFirst(tempQ.left);
            }
            if (tempP.right != null && tempQ.right != null) {
                stackP.addFirst(tempP.right);
                stackQ.addFirst(tempQ.right);
            }
        }
        return true;
    }
    /**
     * 无序数组求中间数
     * 复杂度: O(n)
     * 方法1：使用最小堆，维护一个(n+1)/2的大小，将后半段的元素依次与堆顶对比，小于等于抛弃，大于就替换堆顶，当遍历完这个数组，堆顶就是中位数
     * 方法2：使用快排思想，如果发现得到的index==mid那么刚好满足要求返还
     */
    public static double median1(int[] arr){
        int heapSize = (arr.length+1)/2;
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for(int i=0;i<heapSize;i++){
            minHeap.add(arr[i]);
        }
        //遍历对比
        for(int i=heapSize;i<arr.length;i++){
            if(arr[i]>minHeap.peek()){
                minHeap.poll();
                minHeap.add(arr[i]);
            }
        }
        //中位数取决于数组长度
        if(arr.length%2==1){
            return (double)minHeap.peek();
        }else{
            return (double)((minHeap.poll()+minHeap.peek())/2.0);
        }
    }

    public static double median2(int[] arr){
        //base case
        if(arr==null || arr.length==0){
            return 0;
        }
        int left = 0, right = arr.length-1;
        //标定理论上的中位数位置
        int midIndex = right >> 1;
        int index = -1;
        while (index!=midIndex){
            index = partition(arr,left,right);
            //找到的数字太小，中位数一定在右边
            if(index<midIndex){
                left = index+1;
            }else if(index>midIndex){
                right = index-1;
            }else{
                break;
            }
        }
        return arr[index];
    }
    public static int partition(int[] arr,int left, int right){
        if(left>right){
            return -1;
        }
        //标定值
        int pos = right;
        right--;
        while (left<=right){
            while (left<pos && arr[left]<=arr[pos]){
                left++;
            }
            while (right>pos && arr[right]>=arr[pos]){
                right--;
            }
            if(left>=right){
                break;
            }
            swap(arr,left,right);
        }
        swap(arr,left,pos);
        return left;
    }
    public static void swap(int[] arr, int left, int right){
        int temp = arr[left];
        arr[left] = arr[right];
        arr[right] = temp;
    }
    /**
     * LeetCode 99, 恢复二叉树
     * Level：Hard
     * 已知二叉树的中序排列应该是升序的，如果错误的交换了两个节点，一定会打破顺序
     * 当我们找到了这两个节点后进行交换即可
     */
    public void recoverTree(TreeNode root){
        if(root==null){
            return;
        }
        List<TreeNode> nums = new ArrayList<>();
        //中序排列
        inOrder(root,nums);
        //寻找乱序的两个节点
        TreeNode numOne = null, numTwo = null;
        for(int i=0;i<nums.size()-1;++i){
            //因为中序是升序排列的，所以第一个遇到i是第一个乱序数字
            //第二个乱序数字是i+1
            if(nums.get(i).val>nums.get(i+1).val){
                //让numTwo始终获取第二个乱序数字
                numTwo = nums.get(i+1);
                //numOne是null说明i是第一个乱序数字
                if(numOne==null){
                    numOne = nums.get(i);
                }
            }
        }
        //交换
        if (numOne!=null && numTwo!=null){
            int temp = numOne.val;
            numOne.val = numTwo.val;
            numTwo.val = temp;
        }
    }
    public void inOrder(TreeNode node, List<TreeNode> nums){
        if(node==null){
            return;
        }
        inOrder(node.left,nums);
        nums.add(node);
        inOrder(node.right,nums);
    }
    /**
     * LeetCode 235，儿叉搜索树的最近公共祖先
     * Level: Easy
     * 遍历二叉树，将所有的父节点与子节点成对保存
     * 将p的所有父节点装进set,然后q在set里循环查找自己的父节点
     */
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q){
        Deque<TreeNode> stack = new ArrayDeque<>();
        //父子节点配对<子节点，父节点>
        Map<TreeNode, TreeNode> parent = new HashMap<>();
        parent.put(root,null);
        stack.addFirst(root);
        while(!parent.containsKey(p) || !parent.containsKey(q)){
            TreeNode node = stack.removeFirst();
            //left
            if(node.left!=null){
                parent.put(node.left,node);
                stack.addFirst(node.left);
            }
            //right
            if(node.right!=null){
                parent.put(node.right,node);
                stack.addFirst(node.right);
            }
        }
        //首先将p的所有parent放在set里
        Set<TreeNode> parentSet = new HashSet<>();
        while(p!=null){
            parentSet.add(p);
            //p更新为pD的父节点
            p = parent.get(p);
        }
        //寻找p的父节点集合中，哪个也是q的父节点
        while(!parentSet.contains(q)){
            parentSet.add(q);
            q = parent.get(q);
        }
        return q;
    }
    /**
     * LeetCode 696, 计数二进制子串
     * Level：Easy
     * 我们将连续的0，1统计起来如u,v，去取两个相邻部分的Min(u,v)，这个值就是这两个相邻部分的共享值
     * 例如[00011], Min(2,3) = 2, 也就是说满足条件的子序列有两种：0011，01
     */
    public int countBinarySubstrings(String s){
        List<Integer> countList = new ArrayList<>();
        int ptr = 0, len = s.length();
        while (ptr<len){
            char prevDigit = s.charAt(ptr);
            int count = 0;
            while (ptr<len && s.charAt(ptr)==prevDigit){
                count++;
                ptr++;
            }
            countList.add(count);
        }
        int res = 0;
        for(int i=1;i<countList.size();i++){
            res += Math.min(countList.get(i-1),countList.get(i));
        }
        return res;
    }
    /**
     * LeetCode 1219, 黄金矿工
     * Level: Medium
     * dfs遍历每种路径取最优，因为只能走一次需要进行无效化标注，当路径遍历完之后要重新回复原本值
     */
    public int rows1219, cols1219;
    public int getMaximumGold(int[][] grid){
        rows1219 = grid.length;
        if(rows1219==0){return 0;}
        cols1219 = grid[0].length;
        int res = Integer.MIN_VALUE;
        for(int row=0;row<rows1219;row++){
            for(int col=0;col<cols1219;col++){
                if(grid[row][col]!=0){
                    res = Math.max(res, bfs(grid,row,col));
                }
            }
        }
        return res;
    }
    public int bfs(int[][] grid, int row, int col){
        if(row<0 || row>=rows1219 || col<0 || col>=cols1219 || grid[row][col]==0){
            return 0;
        }
        int gold = grid[row][col];
        //无效化标注
        grid[row][col] = 0;
        int res = Math.max(bfs(grid,row+1,col),Math.max(bfs(grid,row-1,col),Math.max(bfs(grid,row,col+1),bfs(grid,row,col-1))))+gold;
        //遍历需回复至遍历前状态，保证其他路径不会被干扰
        grid[row][col] = gold;
        return res;
    }
    /**
     * LeetCode 1138, 字母板上的路径
     * Level: Medium
     * 因为字母板是有序的，因此横坐标就是 (curChar-'a')/5, 纵坐标就是 (curChar-'a')%5
     * 只有z需要特殊处理，上一个字母是z时需要
     */
    public char[][] alphabetBoard = new char[][]{{'a','b','c','d','e'},
            {'f','g','h','i','j'},
            {'k','l','m','n','o'},
            {'p','q','r','s','t'},
            {'u','v','w','x','y'},
            {'z'}};
    public String alphabetBoardPath(String target){
        StringBuffer sb = new StringBuffer();
        int prevRow = 0, prevCol = 0;
        for(int i=0;i<target.length();i++){
            char digit = target.charAt(i);
            //计算机横纵坐标
            int row = (digit-'a')/5;
            int col = (digit-'a')%5;
            //判断z,需要先向上移动，再左右移动
            if (prevRow==5){
                prevRow = rowHelp(sb,prevRow,row);
                prevCol = colHelp(sb,prevCol,col);
            }else{
                prevCol = colHelp(sb,prevCol,col);
                prevRow = rowHelp(sb,prevRow,row);
            }
            //匹配结束后需要 '!'
            sb.append('!');
        }
        return sb.toString();
    }
    //判断是否需要上下移动U,D
    public int rowHelp(StringBuffer sb, int prevRow, int row){
        if(row!=prevRow){
            int dis = Math.abs(prevRow-row);
            while ((dis--)>0){
                sb.append(row>prevRow? 'D':'U');
            }
        }
        return row;
    }
    //判断是否需要左右移动R,L
    public int colHelp(StringBuffer sb, int prevCol, int col){
        if(prevCol!=col){
            int dis = Math.abs(prevCol-col);
            while ((dis--)>0){
                sb.append(col>prevCol? 'R':'L');
            }
        }
        return col;
    }
    /**
     * LeetCod 128, 最长连续序列
     * Level: Hard
     * 将所有数字塞进一个set, 去暴力尝试每个数字的x, x+1, x+2, ... ,x+y的存在
     * 我们可以通过尝试是否存在x-1的前驱数字，如果存在我们可以跳过，因为以x-1起始的子序列一定才是最长的
     */
    public int longestConsecutive(int[] nums){
        Set<Integer> num_set = new HashSet<>();
        for(int num:nums){
            num_set.add(num);
        }
        int longestStreak = 0;
        for(int num:nums){
            //判断是否有前驱数字
            if(!num_set.contains(num-1)){
                int currNum = num;
                int currStreak = 1;
                while (num_set.contains(currNum+1)){
                    currNum++;
                    currStreak++;
                }
                longestStreak = Math.max(longestStreak,currStreak);
            }
        }
        return longestStreak;
    }
    /**
     * LeeCode 733, 图像渲染
     * Level：Easy
     */
    public int[][] floodFill(int[][] image, int sr, int sc, int newColor){
        bfs(image,sr,sc,newColor,image[sr][sc]);
        return image;
    }
    public void bfs(int[][] image, int sr, int sc, int newColor, int prevColor){
        if(sr<0 || sr>=image.length || sc<0 || sc>=image[0].length || image[sr][sc]!=prevColor || image[sr][sc]==newColor){
            return;
        }
        image[sr][sc] = newColor;
        bfs(image,sr+1,sc,newColor,prevColor);
        bfs(image, sr-1,sc,newColor,prevColor);
        bfs(image,sr,sc+1,newColor,prevColor);
        bfs(image,sr,sc-1,newColor,prevColor);
    }
    /**
     * LeetCode 3, 无重复字符的最长子串
     * Level：Medium
     */
    public int lengthOfLongestSubstring(String s){
        int n = s.length();
        int right = -1;
        int maxLen = 0;
        Set<Character> set = new HashSet<>();
        for(int i=0;i<s.length();i++){
            if(i!=0){
                //当左指针不为0时，一定是上一轮的右指针发现了重复元素，所以需要向右移动左指针并移除最左边元素
                set.remove(s.charAt(i-1));
            }
            while (right+1<n && !set.contains(s.charAt(right+1))){
                //向右移动右指针
                set.add(s.charAt(right+1));
                right++;
            }
            maxLen = Math.max(maxLen,right-i+1);
        }
        return maxLen;
    }
    /**
     * LeetCode 14, 最长公共前缀
     * Level: Easy
     */
    public String longestCommonPrefix(String[] strs){
        if(strs.length==0){
            return "";
        }
        String prefix = strs[0];
        for(String str:strs){
            int index = 0;
            for(;index<str.length()&&index<prefix.length();index++){
                if(str.charAt(index)!=prefix.charAt(index)){
                    break;
                }
            }
            prefix = str.substring(0,index);
            if(prefix.isEmpty()){
                return prefix;
            }
        }
        return prefix;
    }
}
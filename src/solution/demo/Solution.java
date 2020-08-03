package solution.demo;
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
}
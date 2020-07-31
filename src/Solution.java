import javafx.util.Pair;
import org.omg.CORBA.CODESET_INCOMPATIBLE;
import sun.awt.image.ImageWatched;
import sun.reflect.generics.tree.Tree;

import javax.swing.tree.TreeNode;
import java.util.*;

public class Solution {


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
    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

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
    public int integerBreak(int n){
        //当n<=3时，丢弃一个因子，返还n-1
        if(n<=3){
            return n-1;
        }
        //当n>3时，得到商a和余数b, n = 3a + b
        int a = n/3, b = n%3;
        //当b为0时，直接返还3^a即可
        if(b==0){
            return (int) Math.pow(3, a);
        }
        //当b为1时，要将最后一个1+3 转换成 2+2, n = 3^(a-1)*4
        if(b==1){
            return (int) Math.pow(3, a-1) * 4;
        }
        //当b为2时，n = 3^a * 2, 因为2为第二选择, 不需要拆分成1+1
        return (int) Math.pow(3, a) * 2;
    }
}
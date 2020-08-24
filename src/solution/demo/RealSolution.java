package solution.demo;

import javax.script.ScriptContext;
import java.util.*;


public class RealSolution{
//    private static HashMap<Character, Character> map;
//        public static void main(String[] args){
//            Scanner sc = new Scanner(System.in);
//            while (sc.hasNext()){
//                int n = sc.nextInt();
//                long num = 1L;
//                for (int i=1;i<=n;i++){
//                    System.out.println(num);
//                    num = num*i;
//                }
//                String str = String.valueOf(num);
//                int zeroCount = 0;
//                for(int i=str.length()-1;i>=0;i--){
//                    if(str.charAt(i)!='0'){
//                        break;
//                    }
//                    zeroCount++;
//                }
//                System.out.println(zeroCount);
//            }
//    }
//    public static void main(String[] args){
//        Scanner sc = new Scanner(System.in);
//        while (sc.hasNext()){
//            String path = sc.nextLine();
//            boolean[][] grid = new boolean[path.length()*2+1][path.length()*2+1];
//            int row = path.length(), col = path.length();
//            grid[row][col] = true;
//            boolean isFinished = false;
//            for(int i=0;i<path.length();i++){
//                char step = path.charAt(i);
//                if(step=='N'){
//                    row--;
//                    if(grid[row][col]){
//                        isFinished = true;
//                        System.out.println("True");
//                        break;
//                    }else{
//                        grid[row][col] = true;
//                    }
//                }
//                if(step=='S'){
//                    row++;
//                    if(grid[row][col]){
//                        isFinished = true;
//                        System.out.println("True");
//                        break;
//                    }else{
//                        grid[row][col] = true;
//                    }
//                }
//                if(step=='E'){
//                    col++;
//                    if(grid[row][col]){
//                        isFinished = true;
//                        System.out.println("True");
//                        break;
//                    }else{
//                        grid[row][col] = true;
//                    }
//                }
//                if(step=='W'){
//                    col--;
//                    if(grid[row][col]){
//                        isFinished = true;
//                        System.out.println("True");
//                        break;
//                    }else {
//                        grid[row][col] = true;
//                    }
//                }
//            }
//            if(!isFinished){
//                System.out.println("False");
//            }
//        }
//    }
//    public static void main(String[] args){
//        map = new HashMap<>();
//        map.put(')','(');
//        map.put(']','[');
//        map.put('}','{');
//        Scanner sc = new Scanner(System.in);
//        while(sc.hasNext()){
//            char[] chars = sc.nextLine().toCharArray();
//            Stack<Character> stack = new Stack<>();
//            for (char temp : chars) {
//                if (!map.containsKey(temp)) {
//                    stack.push(temp);
//                } else {
//                    char pair = map.get(temp);
//                    if (pair != stack.peek()) {
//                        break;
//                    }
//                    stack.pop();
//                }
//            }
//            if(stack.isEmpty()){
//                System.out.println("True");
//            }else {
//                System.out.println("False");
//            }
//        }
//    }
    //tx
    static class ListNode{
        int val;
        ListNode next;
        ListNode(int val){
            this.val = val;
        }
}
//    public static void main(String[] args){
//        Scanner sc = new Scanner(System.in);
//        while (sc.hasNext()){
//            int n = sc.nextInt();
//            int k = sc.nextInt();
//            for(int i=1;i<=n;i++){
//                int num = sc.nextInt();
//                if (i==k){
//                    continue;
//                }
//                if(i==n){
//                    System.out.print(num);
//                }else{
//                    System.out.print(num+" ");
//                }
//            }
//            ListNode head = new ListNode(0);
//            ListNode temp = head;
//            for(int i=0;i<n;i++){
//                temp.next = new ListNode(sc.nextInt());
//                temp = temp.next;
//            }
//            temp = head.next;
//            ListNode prev = head;
//            int count = 1;
//            while (temp!=null){
//                if(count == k){
//                    prev.next = temp.next;
//                    break;
//                }
//                prev = temp;
//                temp = temp.next;
//                count++;
//            }
//            temp = head.next;
//            while (temp!=null){
//                if(temp.next!=null){
//                    System.out.print(temp.val+" ");
//                }else{
//                    System.out.print(temp.val);
//                }
//                temp = temp.next;
//            }
//        }
//    }
//    public static void main(String[] args){
//        Scanner sc = new Scanner(System.in);
//        while (sc.hasNext()){
//            String str = sc.nextLine();
//            TreeSet<String> set = new TreeSet<>();
//            int k = Integer.parseInt(sc.nextLine());
//            int lptr = 0;
//            while (lptr<=str.length()){
//                for(int rptr=lptr+1;rptr<=str.length();rptr++){
//                    set.add(getSub(str,lptr,rptr));
//                }
//                lptr++;
//            }
//            int count = 1;
//            for(String temp:set){
//                if(count==k){
//                    System.out.println(temp);
//                    break;
//                }
//                count++;
//            }
//        }
//    }
//    public static String getSub(String str, int left, int right){
//        return str.substring(left,right);
//    }
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()){
            String str = sc.nextLine();
            int q = sc.nextInt();
            int[][] pairs = new int[q][2];
            for(int i=0;i<q;i++){
                pairs[i][0] = sc.nextInt();
                pairs[i][1] = sc.nextInt();
            }
            for(int[] pair:pairs){
                String temp = str.substring(pair[0],pair[1]);
                int maxLen = 0;
                for(int i=0;i<temp.length();i++){
                    int len = Math.max(getPaildomic(temp,i,i),getPaildomic(temp,i,i+1));
                    maxLen = Math.max(maxLen,len);
                }
                int count = 1;
                if(maxLen==temp.length()){
                    System.out.println(count);
                }else{
                    temp = temp.substring(maxLen,temp.length());
                    for(int i=maxLen+1;i<temp.length();i++){
                        int len = Math.max(getPaildomic(temp,i,i),getPaildomic(temp,i,i+1));
                        if (len == temp.length()) {
                            count++;
                        }
                    }
                    System.out.println(count);
                }
            }
        }
    }
    public static int getPaildomic(String str, int left, int right){
        while ( left>=0 && right<=str.length() &&str.charAt(left)==str.charAt(right)){
            left--;
            right++;
        }
        return right-left;
    }
}
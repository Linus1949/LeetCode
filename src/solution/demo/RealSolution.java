package solution.demo;
import java.util.*;

public class RealSolution {
    public static HashMap<Character, Character> map = new HashMap<>();
    public static boolean IsValidExp(String s){
        if("".equals(s) || s==null){
            return true;
        }
        if(s.length()==1){
            return false;
        }
        map.put(')','(');
        map.put(']','[');
        map.put('}','{');
        Stack<Character> stack = new Stack<>();
        for(int i=0;i<s.length();i++){
            char temp = s.charAt(i);
            if(!map.containsKey(temp)){
                stack.push(temp);
            }else{
                char pair = stack.pop();
                if(pair!=map.get(temp)){
                    return false;
                }
            }
        }
        return stack.isEmpty();
    }

//    public int GetCoinCount (int N){
//
//    }
//    public int track(int remian, int[] coins){
//        if(remian<0){
//            return -1;
//        }
//        if (remian==0){
//            return 1;
//        }
//        for(int coin:coins){
//            return track(remian-coin,coins)
//        }
//    }
    public static boolean Game24Points(int[] arr) {
        return track(arr,0,arr.length-1,0);
    }
    public static boolean track(int[] arr, int left, int right, int val){
        if(left>right){
            return false;
        }
        if(val==24){
            return true;
        }
        boolean res = false;
        for(int i=left;i<=right;i++){
            res =   track(arr, left+1,right,val+arr[left]) ||
                    track(arr, left+1,right,val-arr[left]) ||
                    track(arr, left+1,right,val*arr[left]) ||
                    track(arr, left+1,right,val/arr[left]);
        }
        return res;
    }
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()){
            int n = sc.nextInt();
            int[] arr = new int[n];
            for(int i=0;i<n;i++){
                arr[i] = sc.nextInt();
            }
            boolean res = Game24Points(arr);
            System.out.println("The 24 is :"+res);
        }
    }
}

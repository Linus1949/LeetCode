package solution.demo;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;


/**
 * @author Linus
 */
public class RealSolution{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()){
            int n = sc.nextInt();
            char[] nums = new char[n];
            List<Character> num5 = new ArrayList<>();
            List<Character> num0 = new ArrayList<>();
            for(int i=0;i<n;i++){
                char num = sc.next().charAt(0);
                if(num=='5'){
                    num5.add(num);
                }else{
                    num0.add(num);
                }
            }
            int i = 0;
            for(Character ch:num5){
                nums[i++] = ch;
            }
            for (Character ch:num0){
                nums[i++] = ch;
            }
            String res = null;
            boolean flag = false;

            for(int j=0;j<num0.size()*num5.size();j++){
                String temp = String.valueOf(nums);
                long num = Long.parseLong(temp);
                if(num%90.0==0){
                    flag = true;
                    res = String.valueOf(nums);
                    break;
                }else{
                    nums = swap(nums);
                }
            }
            if(flag){
                System.out.println(res);
            }else{
                System.out.println(-1);
            }
        }
    }
    public static char[] swap(char[] nums){
        for(int i=0;i<nums.length;i++){
            if(nums[i]=='0' && (i == nums.length - 1 || nums[i + 1] == '0')){
                nums[i-1] = '0';
                nums[i] = '5';
                break;
            }
        }
        return nums;
    }
}
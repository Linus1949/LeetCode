package solution.demo;

import java.math.BigDecimal;
import java.math.MathContext;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.*;

public class RealSolution {
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        while(sc.hasNext()){
            while(sc.hasNext()) {
                int n = sc.nextInt();
                double res = 0L;
                for(int i=1;i<=n;i++){
                    res += ((1.0/(10*i-5)) - (1.0/10*i));
                }
                BigDecimal bigDecimal = new BigDecimal(res);
                System.out.println(bigDecimal.round(new MathContext(4)));
//                int n = sc.nextInt();
//                int m = sc.nextInt();
//                int count = 0;
//                int dis = m-n;
//                for(int i=0;i<=dis;i++){
//                    String num = String.valueOf(n+i);
//                    for(int j=0;j<num.length()-1;j++){
//                        if(isPalindrome(new StringBuffer(num).deleteCharAt(j).toString()) && isPrime(new StringBuffer(num).deleteCharAt(j).toString())){
//                            count++;
//                            break;
//                        }
//                    }
//                }
//                System.out.println(count);
            }
        }
    }
    public static boolean isPalindrome(String num){
        int left = 0, right = num.length()-1;
        if(right==0){
            return true;
        }
        int len = (right-left+1)/2;
        for(int i=0;i<len;i++){
            if(num.charAt(left+i)!=num.charAt(right-i)){
                return false;
            }
        }
        return true;
    }
    public static boolean isPrime(String num){
        int nums = Integer.parseInt(num);
        int count = 0;
        for(int i=1;i<nums;i++){
            if(nums%i==0){
                count++;
            }
        }
        return count == 1;
    }
}

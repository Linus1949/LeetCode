package solution.demo;

import java.util.*;

/**
 * @author wumengyu
 */
public class RealSolution{
//    public static void main(String[] args){
//        Scanner sc = new Scanner(System.in);
//        while (sc.hasNext()){
//            int n = Integer.parseInt(sc.nextLine());
//            int count = 0;
//            for(int i=0;i<n;i++){
//                String name = sc.nextLine().toLowerCase();
//                if(name.length()>10){
//                    continue;
//                }
//                boolean isDigits = true;
//                for(int j=0;j<name.length();j++){
//                    if(name.charAt(j)>='a' && name.charAt(j)<='z'){
//                        continue;
//                    }else{
//                        isDigits = false;
//                        break;
//                    }
//                }
//                if(isDigits){
//                    count++;
//                }
//            }
//            System.out.println(count);
//        }
//    }
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()){
            int n = sc.nextInt();
            int m = sc.nextInt();
            List<Integer> nums = new LinkedList<>();
            for(int i=1;i<=n;i++){
                nums.add(i);
            }
            int[] ops = new int[m];
            int count = 0;
            for(int i=0;i<m;i++){
                ops[i] = sc.nextInt();
                if(ops[i]==2){
                    count++;
                }
            }
            if(count%2==0){
                for(int i=0;i<m;i++){
                    int method = ops[i];
                    if(method==1){
                        nums = methodOne(nums);
                    }
                }
            }else{
                for(int i=0;i<m;i++){
                    int method = ops[i];
                    if(method==1){
                        nums = methodOne(nums);
                    }else{
                        nums = methodTwo(nums);
                    }
                }
            }
            for(int i=0;i<nums.size();i++){
                if(i==nums.size()-1){
                    System.out.print(nums.get(i));
                }else{
                    System.out.print(nums.get(i)+" ");
                }
            }
        }
    }
    public static List<Integer> methodOne(List<Integer> nums){
        int first = nums.remove(0);
        nums.add(nums.size(),first);
        return nums;
    }
    public static List<Integer> methodTwo(List<Integer> nums){
        for(int i=0;i<=nums.size()-1;i=i+2){
            int temp = nums.get(i);
            nums.set(i,nums.get(i+1));
            nums.set(i+1,temp);
        }
        return nums;
    }
}
package solution.demo;
import java.util.*;

public class RealSolution {
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        while(sc.hasNext()){
            while(sc.hasNext()) {
                int n = sc.nextInt();
                int[] nums = new int[n];
                for(int i=0;i<n;i++){
                    nums[i] = sc.nextInt();
                }
                //
                int[] count = new int[n];
                int res = 0;
                for(int i=0;i<n;i++){
                    List<Integer> primes = new ArrayList<>();
                    for(int j=2;j<nums[i];j++){
                        if(isPrime(j)==1){
                            primes.add(j);
                        }
                    }
                    count[i] = Math.max(isPrime(nums[i]),split(nums[i], primes,0));
                    res += count[i];
                }
                System.out.println(res);
            }
        }
    }
    public static int split(int num, List<Integer> primes, int count){
        for(int prime:primes){
            if(num-prime==0){
                break;
            }else if(num-prime<primes.get(0)){
                return 0;
            }else{
                count = Math.max(count, split(num-prime,primes,count+1));
            }
        }
        return count;
    }
    public static int isPrime(int num){
        if(num==1 || num==0){
            return 0;
        }
        if(num==2){
            return 1;
        }
        int count = 1;
        for(int i=2;i<num;i++){
            if(num%i==0){
                count++;
            }
        }
        return count<=1?1:0;
    }
}
package solution.demo;

import java.util.*;

public class RealSolution {
    public static List<int[]> houseList;
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()) {
            int n = sc.nextInt();
            int m = sc.nextInt();

            int[] coins = new int[n];
            for(int i=0;i<n;i++){
                coins[i] = sc.nextInt();
            }
            houseList = new ArrayList<>();
            for(int i=0;i<m;i++){
                houseList.add(new int[]{sc.nextInt(), sc.nextInt()});
            }
            houseList.sort(new Comparator<int[]>() {
                @Override
                public int compare(int[] o1, int[] o2) {
                    return o2[0] - o1[0];
                }
            });

            //do[i]：在i个人的时候最大舒适值
            int[] dp = new int[n+1];
            dp[0] = maxConfort(coins[0]);
            for(int i=1;i<=n;i++){
                dp[i] = dp[i-1] + maxConfort(coins[i]);
            }
            System.out.println(dp[n]);
        }
    }

    public static int maxConfort(int coin){
        int maxConfort = 0;
        for(int[] house: houseList){
            if(house[0]==0){
                continue;
            }
            if(coin>=house[1]){
                maxConfort = house[0];
                //选择后无效化处理
                house[0] = 0;
                break;
            }
        }
        return maxConfort;
    }
//    public static void main(String[] args){
//        Scanner sc = new Scanner(System.in);
//        while(sc.hasNext()){
//            String t = sc.nextLine();
//            int aCount = 0, cCount=0, eCount=0, bCount=0, dCount=0, fCount=0;
//            int leftPtr = 0;
//            int rightPtr = 0;
//            while(ri)
//        }
//    }
}

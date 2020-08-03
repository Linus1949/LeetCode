package solution.demo;

import java.util.*;

public class RealSolution {
//    public static void main(String[] args){
//        Scanner sc = new Scanner(System.in);
//        while (sc.hasNext()){
//            int k = sc.nextInt();
//            int n = sc.nextInt();
//            int[] box = new int[n];
//            for(int i=0;i<n;i++){
//                box[i] = sc.nextInt();
//            }
//            int dis = k;
//            int backSteps = 0;
//            for(int i=0;i<n;i++){
//                if(dis==box[i]){
//                    System.out.println("paradox");
//                    return;
//                }else if(dis>box[i]){
//                    dis -= box[i];
//                }else{
//                    dis = k - (box[i]-dis);
//                    backSteps++;
//                }
//            }
//            System.out.println(dis + " " + backSteps);
//        }
//    }

    public static class box{
        int upSide;
        int downSide;
        int leftSide;
        int rightSide;
        int topSide;
        int bottomSide;

        box(int upSide, int downSide, int leftSide, int rightSide, int topSide, int bottomSide){
            this.upSide = upSide;
            this.downSide = downSide;
            this.leftSide = leftSide;
            this.rightSide = rightSide;
            this.topSide = topSide;
            this.bottomSide = bottomSide;
        }
    }
//    public static void main(String[] args){
//        Scanner sc = new Scanner(System.in);
//        while (sc.hasNext()){
//            int n = sc.nextInt();
//            List<box> boxes = new ArrayList<>();
//            for(int i=0;i<n;i++){
//                boxes.add(new box(sc.nextInt(), sc.nextInt(), sc.nextInt(), sc.nextInt(), sc.nextInt(), sc.nextInt()));
//            }
//            HashMap<>
//
//        }
//    }
    public static class Daymeal{
        int cal;
        int deal;

        Daymeal(int cal, int deal){
            this.cal = cal;
            this.deal = deal;
        }
    }
    public static class Nightmeal{
        int cal;
        int deal;

        Nightmeal(int cal, int deal){
            this.cal = cal;
            this.deal = deal;
        }
    }
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()){
            int n = sc.nextInt();
            int m = sc.nextInt();
            int t = sc.nextInt();
            List<Daymeal> DM = new ArrayList<>();
            List<Nightmeal> NM = new ArrayList<>();
            for(int i=0;i<n;i++){
                DM.add(new Daymeal(sc.nextInt(), sc.nextInt()));
            }
            for(int i=0;i<m;i++){
                NM.add(new Nightmeal(sc.nextInt(), sc.nextInt()));
            }
            DM.sort(Comparator.comparingInt(o -> o.cal));
            NM.sort((Comparator.comparingInt(o -> o.cal)));
            //遍历中餐
            if(t==0){
                System.out.println(0);
            }
            int meals = Integer.MAX_VALUE;
            for(Daymeal dm:DM){
                if(dm.deal>=t){
                    meals = Math.min(meals, dm.cal);
                    break;
                }else{
                    meals = Math.min(meals, dm.cal+track(NM, t-dm.deal));
                }
            }
            if(meals<0){
                System.out.println(-1);
                return;
            }
            System.out.println(meals);
        }
    }
    public static int track(List<Nightmeal> meals, int deals){
        int res = Integer.MAX_VALUE;
        int count = 0;
        for(Nightmeal temp:meals){
            if(temp.deal>=deals) {
                res = Math.min(res, temp.cal);
                break;
            }
            else{
                count++;
            }
        }
        if(count>=meals.size()){
            return Integer.MIN_VALUE;
        }
        return res;
    }
}

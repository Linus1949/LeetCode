package solution.demo;

import javafx.util.Pair;
import org.omg.PortableInterceptor.INACTIVE;

import java.util.*;

/**
 * @author wumengyu
 */
public class RealSolution{
    public static String compress(String str){
//        HashMap<Character,Integer> map = new HashMap<>();
//        for(Character ch:str.toCharArray()){
//            if(!map.containsKey(ch)){
//                map.put(ch,1);
//            }else{
//                map.put(ch, map.get(ch)+1);
//            }
//        }
//        StringBuilder sb = new StringBuilder();
//        for(Character ch:map.keySet()){
//            sb.append(ch);
//            sb.append(map.get(ch));
//        }
//        String res = sb.toString();
//        return (res.length()<str.length())? res:str;
        StringBuilder sb = new StringBuilder();
        int count = 1;
        for(int i=0;i<str.length()-1;i++){
            if(str.charAt(i)!=str.charAt(i+1)){
                sb.append(str.charAt(i));
                sb.append(count);
                count = 1;
            }else{
                count++;
            }
        }
        sb.append(str.charAt(str.length()-1));
        sb.append(count);
        String res = sb.toString();
        return (res.length()<str.length())? res:str;
    }

    public int[][] convert(int[][] matrix){
        for(int i=0;i< matrix.length/2;i++){
            int[] temp = matrix[i];
            matrix[i] = matrix[(matrix.length-1)-i];
            matrix[(matrix.length-1)-i] = temp;
        }
        return matrix;
    }
    class City{
        int cityNum;
        List<Pair<City,Integer>> paths;
        public City(int cityNum){
            this.cityNum = cityNum;
        }
    }
    public int[] findAllCheapestPrice (int n, int[][] flights, int src) {
        List<City> cityList = new ArrayList<>();
        for(int i=0;i<n;i++){
            City temp = new City(i);
            //默认到自己的距离为0
            temp.paths.add(new Pair<>(temp,0));
        }
        for(int[] trip:flights){
            int start = trip[0];
            int end = trip[1];
            int dis = trip[2];
            City endCity = cityList.get(end);
            City startCity = cityList.get(start);
            startCity.paths.add(new Pair<>(endCity,dis));
        }
        //返回数组
        int[] res = new int[n];
        City target = cityList.get(src);
        for (int i=0;i<n;i++){
            List<Pair<City,Integer>> pairList = target.paths;
            for(Pair<City,Integer> pair:pairList){
                if(pair.getKey().cityNum==i){
                    res[i] = pair.getValue();
                }
            }
        }
        return res;
    }

    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        while (sc.hasNext()){
            String str = sc.nextLine();
            System.out.println(compress(str));
        }
    }
}


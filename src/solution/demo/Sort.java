package solution.demo;

import java.util.Arrays;

public class Sort {
    /**
     * 快排
     * 复杂度：O(nlogn)
     */
    public static class QuickSort{
        private static int count;
        /**
         * 测试
         */
        public static void main(String[] args){
            int[] num = {3,45,78,64,52,11,64,55,99,11,18};
            System.out.println();
        }

        /**
         * 快排
         * @param num 排序数组
         * @param left 数组前指针
         * @param right 数组后指针
         */
        private static void QuickSort(int[] num, int left, int right){
            //如果left等于right,既数组里只有一个元素，直接返回
            if(left>=right){
                return;
            }
            //设置最左边元素为基准值
            int key = num[left];
            //数组中比key小的放在左边，比key大的放在右边，key值下表为i
            int i = left;
            int j = right;
            while(i<j){
                //j向左移动，直到遇到比key小的值
                while (num[j]>=key && i<j){
                    j--;
                }
                //i向右移，直到遇到比key大的
                while(num[i]<=key && i<j){
                    i++;
                }
                //i和j指向的元素交换
                if(i<j){
                    int temp = num[i];
                    num[i] = num[j];
                    num[j] = temp;
                }
            }
            num[left] = num[i];
            num[i] = key;
            count++;
            QuickSort(num, left, i-1);
            QuickSort(num,i+1, right);
        }

        /**
         * 将一个int类型数组转化为字符串
         * @param arr
         * @param flag
         * @return
         */
        private static String arrayToString(int[] arr,String flag) {
            String str = "数组为("+flag+")：";
            for(int a : arr) {
                str += a + "\t";
            }
            return str;
        }

    }

    /**
     * 并归排序
     * 时间复杂度：O(nlongn)
     */
    public static class MergeSort{
        public static void merge(int[] arr, int low, int mid, int high){
            int[] temp = new int[high-low + 1];
            //左指针
            int i = low;
            //右指针
            int j = high;
            int k= 0;
            //把较小的数移动到新数组中
            while(i<=mid && j<=high){
                if(arr[i]<arr[j]){
                    temp[k++] = arr[i++];
                }else{
                    temp[k++] = arr[j++];
                }
            }
            //把左边剩余的数移入数组
            while(i<=mid){
                temp[k++] = arr[i++];
            }
            //把右边的剩余的数一如数组
            while(j<=high){
                temp[k++] = arr[j++];
            }
            //把新数组中的数覆盖num数组
            for(int k2=0;k2<temp.length;k2++){
                arr[k2+low] = temp[k2];
            }
        }

        public static void mergeSort(int[] arr, int low, int high){
            int mid = (low+high)/2;
            if(low<high){
                //左边
                mergeSort(arr, low, mid);
                //右边
                mergeSort(arr, mid+1, high);
                System.out.println(Arrays.toString(arr));
            }
        }
        public static void main(String[] args) {
            int a[] = { 51, 46, 20, 18, 65, 97, 82, 30, 77, 50 };
            mergeSort(a, 0, a.length - 1);
            System.out.println("排序结果：" + Arrays.toString(a));
        }
    }
}

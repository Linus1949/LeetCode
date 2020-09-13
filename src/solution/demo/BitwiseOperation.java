package solution.demo;

/**
 * @author Linus
 * 位运算实现加减乘除以及解决相应算法问题
 */
public class BitwiseOperation {
    /**
     * 先通过异或运算得到不考虑进位的相加，因为XOR会将两个处理为0，不同的处理为1
     * 再通过与运算和左移1一位来获取进位，然后再循环前面两步直至进位项为0
     */
    private int add(int num1, int num2){
        int sum = num1 ^ num2;
        int carry = (num1 & num2) << 1;
        while (carry!=0){
            int a = sum;
            int b = carry;
            sum = a^b;
            carry = (a & b) << 1;
        }
        return sum;
    }
    /**
     * 我们通过加法来实现减法，a+(-b)，要得到-b可以通过取反加1
     */
    private int substract(int num1, int num2){
        //取反加1
        int subtracts = add(~num2,1);
        //add加上负数即可
        int result = add(num1,subtracts);
        return result;
    }
    /**
     * 对于乘法来说，我们可以参考十进制的过程，为0时记0，为1时乘数左移一位
     */
    public int multiply(int num1, int num2){
        //将乘数与被乘数取绝对值
        int multiplicand = num1<0? add(~num1,1): num1;
        int multiplier = num2<0? add(~num2,1): num2;
        //计算绝对值的乘积
        int product = 0;
        while (multiplier>0){
            //每次考察乘数的最后一位
            if((multiplier&0x1)>0){
                //为1时，其实是又取了被乘数一遍
                product = add(product,multiplicand);
            }
            //每一位被乘了一后要向左移一位
            multiplicand = multiplicand << 1;
            //下一轮需要乘数向右移一位再处理
            multiplier = multiplier >> 1;
        }
        //取符号
        if((num1^num2)<0){
            product = add(~product,1);
        }
        return product;
    }
    /**
     * 对于除法运算，
     */
    public int divide(int num1, int num2){
        //取被除数与除数的绝对值
        int dividend = num1<0? add(~num1,1): num1;
        int divisor = num2<0? add(~num2,1): num2;
        int quotient = 0, remainder = 0;
        //我们从2的31次方去尝试
        for(int i=31;i>=0;i--){
            //比较dividend是否大于divisor的(1<<i)次方，不要将dividend与(divisor<<i)比较，而是用(dividend>>i)与divisor比较，
            //效果一样，但是可以避免因(divisor<<i)操作可能导致的溢出，如果溢出则会可能dividend本身小于divisor，但是溢出导致dividend大于divisor
            if((dividend>>i) >= divisor){
                quotient = add(quotient,1<<i);
                dividend = substract(dividend,divisor<<i);
            }
        }
        //确定符号
        if((num1^num2)<0){
            //如果除数与被除数异号，商取反
            quotient = add(~quotient,1);
        }
        //确定余数
        remainder = num2>0? dividend: add(~dividend,1);
        return quotient;
    }
}

class PrimeAtLeast {

    static boolean checkNotPrime(int a) {
        if (a <= 1) {
            return true;
        }
        for (int i = 2; i < a; i++) {
            int remainder = a % i;
            if (remainder == 0) {
                return true;
            }
        }
        return false;
    }

    public static void main(String[] args) {
        if (args.length < 1) {
            return;
        }
        int n = Integer.parseInt(args[0]);
        int p = n;
        for (int i = n; checkNotPrime(i); i++) {
            p++;
        }
        System.out.println("The next prime after " + n + " is: " + p);
    }


    // public static void main(String[] args) {
    //     int n = Integer.parseInt(args[0]);
    //     if (checkNotPrime(n)) {
    //         System.out.println("Not a Prime!");
    //     }
    //     else {
    //         System.out.println("A prime!");
    //     }
    // }
}

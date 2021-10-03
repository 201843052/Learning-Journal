class CharCountA {
    static final int LINE_WIDTH = 80;

    public static void main(String[] args) {
        if (args.length < 1) {
            return;
        }
        int length = args[0].length();
        int lines = ((length - 1) / LINE_WIDTH) + 1;
        System.out.println("has " + length + " characters");
        System.out.println("needs " + lines + " lines");
    }
}

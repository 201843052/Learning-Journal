public class ManyTimes {
    public static void main(String[] args) {
        String s;
        s = args[0];
        s = s + " " + s + " " + s;
        System.out.println(s);
        s = s + " " + s + " " + s;
        System.out.println(s);
        s = s + " " + args[0];
        System.out.println(s);
    }
}

public class Main {
  static {
        System.loadLibrary("native");
    }

  public static void main(String[] args) {
    System.out.println("Hello World");
  }
  public native void sayHello();

}

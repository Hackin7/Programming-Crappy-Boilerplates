import java.util.Scanner; // Input

public class Test
{
	public static void main(String[] args) {
	System.out.println("Java Test Program");
	conditionals();
	loops();
	input();
	}
	static void conditionals(){
	    System.out.println("Conditionals");
	    System.out.println("if, else if, else structure");
	    Scanner input = new Scanner(System.in);
	    int option = input.nextInt();
	    if (option == 1){
	        System.out.println("if statement run ");
	    }
	    else if(option == 2){
	        System.out.println("else if statement run");
	    }
	    else{
	        System.out.println("else statement run");
	    }
	}
	static void loops(){
	    System.out.println("Loops");
	    for (int x = 1; x <=5; x++){
	        System.out.println("for loop when x = "+x);
	    }
	    int y = 1;
        while(y <= 5) { //Checks condition then does things
          System.out.println("while loop when y = "+y);
          y++;
        }
	    int z = 1;
        do { //Does thigns first then checks condition
          System.out.println("do while loop when z = "+z);
          z++;
        } while(z <= 5);
	}
}

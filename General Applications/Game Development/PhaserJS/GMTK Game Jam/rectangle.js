


class GameObject{
	create(context, x, y){}
	update(context, cursors){}
}
class Rectangle extends GameObject{
  constructor(colour, width, height) {
	super();
    this.colour = colour;
    this.width = width;
    this.height = height;
  }
  
  create(context, x, y){
	this.object = context.add.rectangle(x, y, this.height, this.width, this.colour);
	this.object.parent = this;
	this.object.setInteractive(); 
  }
  
  outline(width, colour){
	  this.object.setStrokeStyle(width, colour);
	  //console.log("Stroke Style");
  }
}

class JoinableNode extends Rectangle {
  constructor(size, state) {
    let colour = 0xffffff;
    switch(state){
		  case 1:
		      colour = 0xffffff;
              break;
		  case 2:
		      colour = 0xffff00;
			  break;
		  case 3:
		      colour = 0x00ff00;
              break;
		  case 4:
		      colour = 0xff0000;
              break;
		  default:
              break;
		  
	  } 
    super(colour, size, size);
    this.state = state; //1 for platform, 2 for player, 3 for goal, 4 for death 
  }
  
  create(context, x, y){
	  super.create(context, x, y);
	  //this.underline(4,0xff0000);
	  context.physics.add.existing(this.object);
	  let obj = this.object;
	  this.object.on('pointerdown', function (pointer) {
	    console.log('pointerdown'); 
			this.setStrokeStyle(4,0xff0000);
	  });
	  this.object.on('pointerup', function (pointer) {
			this.setStrokeStyle(0 ,0xff0000);

	  });
	 
  }
  
  update(context, cursors){
	 switch(this.state){
		  case 1:
              //this.object.setPushable(false);
              this.object.body.setImmovable(true);
              this.object.body.setVelocityX(0);
              this.object.body.setVelocityY(0);
              break;
		  case 2:
				//this.object.body.static = false;
			    this.object.body.gravity.y = 400;
			  
			    if (cursors.left.isDown){
					this.object.body.setVelocityX(-160);
				}else if (cursors.right.isDown){
					this.object.body.setVelocityX(160);
				}else{
					this.object.body.setVelocityX(0);
				}

				if (cursors.up.isDown && this.object.body.touching.down)
				{
					this.object.body.setVelocityY(-330);
				}
				
              break;
		  case 3:
              break;
		  default:
              break;
		  
	  } 
  }
}


function joinNodes(node1, node2){
	
}


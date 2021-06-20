class Joiner{
    constructor(context){
        this.queue = [];
        this.context = context;
    }
    add(node){
        this.queue.push(node);
        if (this.queue.length >= 2){
            let i=0, j=1;
            this.queue[i].mouseSelect(()=>null);
            this.queue[j].mouseSelect(()=>null);
            console.log("connect");
            this.queue[i].join(this.queue[j]);
            this.queue[j].join(this.queue[i]);


            let line = this.context.add.line(
                60+30, 0,
                this.queue[i].getMidX(), this.queue[i].getMidY(),
                this.queue[j].getMidX(), this.queue[j].getMidY(),
                0xffffff);
            //this.context.physics.add.existing(line);
            //platforms.add(line);
            //line.body.setImmovable(true);
            //line.body.setSize(100, 100, 0, 0);

            this.queue = [];
        }
    }
}


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

  update(context, cursors){
      if (this.object === null){
          create(context, 0, 0);
      }
  }
  outline(width, colour){
	  this.object.setStrokeStyle(width, colour);
	  //console.log("Stroke Style");
  }
}

class JoinableNode extends Rectangle {
  constructor(size, state, joiner) {
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
    this.pastClicked = 0;
    this.selected = 0;
    this.joiner = joiner;
  }

  mouseSelect(handler){
      this.selected = !this.selected;
      this.outline(this.selected * 4,0xff0000);
      if (this.selected){
         handler(this);
      }else{
      }
  }

  getMidX(){
      let value = this.object.x + this.width / 2;
      console.log(value);
      return this.object.x;
  }
  getMidY(){
      return this.object.y;// + this.height / 2;
  }
  // Phaser Methods /////////////////////////////////////////////////////////////
  create(context, x, y){
	  super.create(context, x, y);
	  //
	  context.physics.add.existing(this.object);
	  let obj = this;
	  this.object.on('pointerdown', function (pointer) {
          if (!this.parent.pastClicked){
              this.parent.pastClicked = 1;
              this.parent.mouseSelect((context)=> {context.joiner.add(obj);});

          }
	  });
	  this.object.on('pointerup', function (pointer) {
          if (this.parent.pastClicked){
              this.parent.pastClicked = 0;
          }

	  });

  }

  update(context, cursors){
      super.update(context, cursors)
	 switch(this.state){
		  case 1:
              //this.object.setPushable(false);
              this.object.body.setImmovable(true);
              this.object.body.setVelocityX(0);
              this.object.body.setVelocityY(0);
              break;
		  case 2:
				//this.object.body.static = false;
			    this.object.body.setImmovable(true);
			    this.object.body.setVelocityX(this.player.object.body.velocity.x);
                this.object.body.setVelocityY(this.player.object.body.velocity.y);
                if (this.object.body.touching.down){
                    this.object.body.setVelocityY(0);
                }
              break;
		  case 3:
                this.object.body.setImmovable(true);
                 break;
          case 4:
              this.object.body.setImmovable(true);
              break;
		  default:
              break;

	  }
  }
  join(node){
      this.state = node.state;
      switch (node.state){
          case 2:
            this.player = node;
            break;
      }
  }
}

function cc(dict){
    return dict.up || dict.left || dict.right || dict.down;
}
class Player extends JoinableNode{
    update(context, cursors){
	    this.object.body.gravity.y = 400;

	    if (cursors.left.isDown){
			this.object.body.setVelocityX(-160);
		}else if (cursors.right.isDown){
			this.object.body.setVelocityX(160);
		}else{
			this.object.body.setVelocityX(0);
		}

        if (this.object.body.touching.down || this.player.object.body.touching.down
            //|| cc(this.object.body.checkCollision)
            //|| cc(this.player.object.body.checkCollision)
        ){
            //this.object.body.x = this.object.body.x;
            this.object.body.setVelocityY(0);
            if (cursors.up.isDown ){
                this.object.body.setVelocityY(-330);
            }

        }

    }
}

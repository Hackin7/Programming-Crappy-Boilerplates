var config = {
    type: Phaser.AUTO,
    width: 800,
    height: 600,
    physics: {
        default: 'arcade',
        arcade: {
            gravity: { y: 0 },//300 },
            debug: false
        }
    },
    scene: {
        preload: preload,
        create: create,
        update: update
    }
};

var height = 10;
var width = 10;
var nodeSize = 60;
var map = [
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 2, 0, 0, 0, 2, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

var mapObjects = [];

function preload ()
{
    //this.load.image('sky', 'assets/sky.png');
    //this.load.image('ground', 'assets/platform.png');
    //this.load.image('star', 'assets/star.png');
    //this.load.image('bomb', 'assets/bomb.png');
    //this.load.spritesheet('dude', 'assets/dude.png', { frameWidth: 32, frameHeight: 48 });
}

var cursors;
var keyObj;

function create ()
{
	
    for (let j=0; j<height; j++){
		for (let i=0; i<width; i++){
			let x = i*nodeSize; let y = j * nodeSize;
			
			let node = null;
			if (map[j][i] !== 0){
				node = new JoinableNode(nodeSize, map[j][i]);
				node.create(this, x, y);
				mapObjects.push(node);
			}
		}
	}
	for(let i=1; i<mapObjects.length; i++){
		for(let j=0;j<i;j++){
			//console.log(mapObjects[i].object, mapObjects[j].object);
			this.physics.add.collider(mapObjects[i].object, mapObjects[j].object);
		}
	}
	
	cursors = this.input.keyboard.createCursorKeys();
  let line1 = this.add.line(0,0,100, 200, 200, 100, 0xffffff);
  this.physics.add.existing(line1);
  line1.body.gravity.y=300;
 	
}

function update(){
	mapObjects.forEach(object => {
		object.update(this, cursors);
	});
	
	if ( this.input.keyboard.addKey('R').isDown){
	  //this.restart();
	}
}


var game = new Phaser.Game(config);


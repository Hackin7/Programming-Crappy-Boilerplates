
var platforms, death, goals;
var height = 10;
var width = 13;
var nodeSize = 60;
var map = [
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 2, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 4, 0, 3],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

var mapObjects = [];
var cursors;
var keyObj;
class Map extends Phaser.Scene{

    constructor ()
    {
        super('Name');
    }

    preload ()
    {
        this.keys = {}
        //this.load.image('sky', 'assets/sky.png');
        //this.load.image('ground', 'assets/platform.png');
        //this.load.image('star', 'assets/star.png');
        //this.load.image('bomb', 'assets/bomb.png');
        //this.load.spritesheet('dude', 'assets/dude.png', { frameWidth: 32, frameHeight: 48 });
    }



    create ()
    {
        this.joiner = new Joiner(this);
        let player = null;
        for (let j=0; j<height; j++){
    		for (let i=0; i<width; i++){
    			let x = i*nodeSize; let y = j * nodeSize;

    			let node = null;
    			if (1 <= map[j][i] && map[j][i] <= 4){
    				node = new JoinableNode(nodeSize, map[j][i], this.joiner);
    				node.create(this, x, y);
    				mapObjects.push(node);
    			}else if (5 <= map[j][i]){//} && map[j][i] <= 4){
                    player = new Player(nodeSize, 2, this.joiner);
    				player.create(this, x, y);
    				mapObjects.push(player);
                }
    		}

            //this.drawKeyboard();
            this.keys.R = this.input.keyboard.addKey(Phaser.Input.Keyboard.KeyCodes.R);
    	}

        cursors = this.input.keyboard.createCursorKeys();
      let line = this.add.line(100,0,100, 200, 200, 100, 0xffffff);
      this.physics.add.existing(line);
      //line1.body.setImmovable(true);
      line.body.gravity.y=100;
      line.body.setSize(100, 100, 0, 0);
      line.body.setCollideWorldBounds(true);


        platforms = this.physics.add.group();
        //platforms.setImmovable(true);
        //platforms.add(line1);
        this.physics.collide(platforms);
        death = this.physics.add.group()
        goals = this.physics.add.group()
        let r = this.restart, m = this.moveon;
        this.physics.collide(platforms, death,
            (n1, n2) => {r();}, null, this);
        this.physics.collide(platforms, goals,
            (n1, n2) => {m();}, null, this);


        console.log(this.physics);
    	for(let i=0; i<mapObjects.length; i++){
            platforms.add(mapObjects[i].object);
            switch(mapObjects[i]){
                case 3:goals.add(mapObjects[i].object);break;
                case 4:goals.add(mapObjects[i].object);break;
            }

            //mapObjects[i].object.body.setCollideWorldBounds(true);
    		for(let j=i+1;j<mapObjects.length;j++){
                this.physics.collide(mapObjects[i], mapObjects[j],
                    (n1, n2) => {console.log("1");r();}, null, this);
                if (mapObjects[i].state == 2 && mapObjects[j].state == 2){
                    mapObjects[i].join(mapObjects[j]);
                    mapObjects[j].join(mapObjects[i]);
                }
    		}
    	}

        this.physics.add.collider(platforms);



    }

    update(){
    	mapObjects.forEach(object => {
    		object.update(this, cursors);
    	});

    	if ( this.keys.R.isDown){
            console.log("Refsresh");
            //this.registry.destroy();
            //this.events.off();
            this.restart();
            //this.scene.start('Name');
    	}

        for(let i=1; i<mapObjects.length; i++){
    		for(let j=0;j<i;j++){
                if (mapObjects[i].selected && mapObjects[j].selected){


                }
    		}
    	}
    }

    restart(){
        this.scene.restart();
        mapObjects = [];
    }

    moveon(){
        console.log("Success!");
        this.scene.start('Name');
        mapObjects = [];
    }

}


var game = new Phaser.Game({
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
    scene: [Map,]
});

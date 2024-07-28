document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('neuralNetworkCanvas');
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const nodes = [];
    const connections = [];

    class Node {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.radius = Math.random() * 2 + 1;
            this.speed = Math.random() * 0.5 + 0.1;
            this.directionX = Math.random() * 2 - 1;
            this.directionY = Math.random() * 2 - 1;
            this.color = `hsl(${Math.random() * 60 + 180}, 100%, 50%)`;
        }

        update() {
            this.x += this.directionX * this.speed;
            this.y += this.directionY * this.speed;

            if (this.x < 0 || this.x > canvas.width) this.directionX *= -1;
            if (this.y < 0 || this.y > canvas.height) this.directionY *= -1;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = this.color;
            ctx.fill();
        }
    }

    class Connection {
        constructor(startNode, endNode) {
            this.startNode = startNode;
            this.endNode = endNode;
            this.lifetime = 0;
            this.maxLifetime = Math.random() * 100 + 50;
        }

        update() {
            this.lifetime++;
        }

        draw() {
            const alpha = Math.sin((this.lifetime / this.maxLifetime) * Math.PI);
            ctx.beginPath();
            ctx.moveTo(this.startNode.x, this.startNode.y);
            ctx.lineTo(this.endNode.x, this.endNode.y);
            ctx.strokeStyle = `rgba(255, 255, 255, ${alpha * 0.5})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
        }
    }

    function init() {
        for (let i = 0; i < 100; i++) {
            nodes.push(new Node());
        }
    }

    function connectNodes() {
        connections.length = 0;
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                if (Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y) < 100) {
                    connections.push(new Connection(nodes[i], nodes[j]));
                }
            }
        }
    }

    function animate() {
        requestAnimationFrame(animate);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        nodes.forEach(node => {
            node.update();
            node.draw();
        });

        connections.forEach((connection, index) => {
            connection.update();
            connection.draw();
            if (connection.lifetime > connection.maxLifetime) {
                connections.splice(index, 1);
            }
        });

        if (Math.random() > 0.98) connectNodes();
    }

    init();
    connectNodes();
    animate();

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        init();
        connectNodes();
    });
});
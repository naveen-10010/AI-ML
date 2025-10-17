inputs=[1,2,3,2.5];
'''weights1=[0.2,0.8,-0.5,1.0];
weights2=[0.5,-0.91,0.26,-0.5];
weights3=[-0.26,-0.27,0.17,0.87];
bais1=2;
bais2=3;
bais3=0.5;
output=[inputs[0]*weights1[0]+inputs[1]*weights1[1]+inputs[2]*weights1[2]+inputs[3]*weights1[3]+bais1,inputs[0]*weights2[0]+inputs[1]*weights2[1]+inputs[2]*weights2[2]+inputs[3]*weights2[3]+bais2,inputs[0]*weights3[0]+inputs[1]*weights3[1]+inputs[2]*weights3[2]+inputs[3]*weights3[3]+bais3];
print(output);'''
ws=[[0.2,0.8,-0.5,1.0],[0.5,-0.91,0.26,-0.5],[-0.26,-0.27,0.17,0.87]];
b=[2,3,0.5];
op=[];
for w, nb in zip(ws,b):
    ot=0;
    for ni, nw in zip(inputs,w):
        ot=ot+(ni*nw);
    ot=ot+nb;
    op.append(ot);
print(op);
        

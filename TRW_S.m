close all
clear all
% I=imread('0_predict_v4.png');
maindir = 'E:\EvaluationTool\map\my12\CVC-3001\'
filename=dir(maindir);
for maskname = 3:length(filename)
    length(filename)
    filename(maskname).name
                    I=imread(['E:\EvaluationTool\map\my12\CVC-3001\',filename(maskname).name]);
                    % I1=im2bw(I);
                    subplot(2,2,1);
                    imshow(I);
                    title('UNET输出图像');
                    J = imread(['E:\EvaluationTool\map\my12\CVC-3001\',filename(maskname).name]);
                    m_answer=zeros(size(I));    %最终的输出结果
                    [m_height,m_width]=size(m_answer);
                    m_nPixels = m_width*m_height;
                    m_messages=zeros(1,4*m_nPixels);   %储存信息的矩阵
                    m_DBinary=zeros(1,m_nPixels);      %一元惩罚项
                    m_horzWeightsBinary = zeros(1,m_nPixels);   %二元水平惩罚项
                    m_vertWeightsBinary = zeros(1,m_nPixels);   %二元竖直惩罚项
                    nLabel = 2;  
                    % I=I';
                    m_D = zeros(1,nLabel*m_nPixels);%m_D是一维的
                    A = reshape(I',m_nPixels,1);
                    B = reshape(J',m_nPixels,1);
                    A = A';
                    B = B';
                    for i = 1:m_nPixels     %给一元项赋值
                        for j = 1:2
                            if j==1
                               m_D((i-1)*nLabel+j)=(A(i)-(255-B(i)))^2;
                            end
                            if j==2
                               m_D((i-1)*nLabel+j)=(((255-A(i))-B(i)))^2;
                            end
                        end
                    end
                    m_V = [0,0.6,0.6,0];
                    for i=1:m_nPixels
                        m_DBinary(i) = m_D(2*i)-m_D(2*i-1);%
                        m_horzWeightsBinary(i) = m_V(2);
                        m_vertWeightsBinary(i) = m_V(2);
                    end 

                    nIterations = 100;
                    Di = 0;
                    while nIterations>0
                        n = 1;
                        M_ptr = 1;
                        for y=1:m_height
                            for x=1:m_width
                               Di = m_DBinary(n);
                               if x>1 
                                   Di = Di+m_messages(M_ptr-2);
                               end
                               if y>1
                                    Di = Di+m_messages(M_ptr-2*m_width+1);
                               end
                               if x<m_width
                                   Di = Di+m_messages(M_ptr);
                               end
                               if y<m_height
                                    Di = Di+m_messages( M_ptr+1);
                               end
                               DiScaled = Di*0.5;
                               if x<m_width
                                   Di = DiScaled-m_messages(M_ptr);
                                   lambda = m_horzWeightsBinary(n);
                                   if lambda<0
                                       Di = -Di;
                                       lambda = -lambda;
                                   end
                                   if Di>lambda
                                       m_messages(M_ptr) = lambda;
                                   else
                                       m_messages(M_ptr) = ternaryOperator(Di<-lambda,-lambda,Di);
                                   end
                               end
                               if y<m_height
                                   Di = DiScaled-m_messages(M_ptr+1);
                                   lambda = m_vertWeightsBinary(n);
                                   if lambda<0
                                       Di = -Di;
                                       lambda = -lambda;
                                   end
                                   if Di>lambda
                                        m_messages(M_ptr+1) = lambda;
                                   else
                                       m_messages(M_ptr+1) = ternaryOperator(Di<-lambda,-lambda,Di);
                                   end
                               end
                               n = n+1;
                               M_ptr = M_ptr+2;
                            end
                        end
                    %back
                        n = n-1;
                        M_ptr = M_ptr-2;
                        for y=m_height:-1:1
                            for x=m_width:-1:1
                               Di = m_DBinary(n);
                               if x>1 
                                   Di = Di+m_messages(M_ptr-2);
                               end
                               if y>1
                                    Di = Di+m_messages(M_ptr-2*m_width+1);
                               end
                               if x<m_width
                                   Di = Di+m_messages(M_ptr);
                               end
                               if y<m_height
                                    Di = Di+m_messages( M_ptr+1);
                               end
                               DiScaled = Di*0.5;
                               if x>1
                                   Di = DiScaled-m_messages(M_ptr-2);
                                   lambda = m_horzWeightsBinary(n-1);
                                   if lambda<0
                                       Di = -Di;
                                       lambda = -lambda;
                                   end
                                   if Di>lambda
                                       m_messages(M_ptr-2) = lambda;
                                   else
                                       m_messages(M_ptr-2) = ternaryOperator(Di<-lambda,-lambda,Di);
                                   end
                               end
                               if y>1
                                   Di = DiScaled-m_messages(M_ptr-2*m_width+1);
                                   lambda = m_vertWeightsBinary(n-m_width);
                                   if lambda<0
                                       Di = -Di;
                                       lambda = -lambda;
                                   end
                                   if Di>lambda
                                        m_messages(M_ptr-2*m_width+1) = lambda;
                                   else
                                       m_messages(M_ptr-2*m_width+1) = ternaryOperator(Di<-lambda,-lambda,Di);
                                   end
                               end
                               n = n-1;
                               M_ptr = M_ptr-2;
                            end
                        end
                        nIterations=nIterations-1;
                    end
                    M_ptr = 1;
                    n = 1;
                    size(m_answer);
                    for y = 1:m_height
                        for x = 1:m_width
                            Di = m_DBinary(n);
                            if x>1
                                Di = Di+ ternaryOperator(m_answer(n-1)==0,m_horzWeightsBinary(n-1),-m_horzWeightsBinary(n-1));
                            end
                            if y>1
                                Di = Di+ ternaryOperator(m_answer(n-m_width)==0,m_vertWeightsBinary(n-m_width),-m_vertWeightsBinary(n-m_width));
                            end
                            if x<m_width
                                Di = Di+m_messages(M_ptr);
                            end
                            if y<m_height
                                Di = Di+m_messages(M_ptr+1);
                            end
                            %%%%min
                            m_answer(y,x) = ternaryOperator(Di>=0,0,1);%matlab总是先竖向优先
                            M_ptr = M_ptr+2;
                            n = n+1;
                        end
                    end
                    subplot(2,2,3);
                    size(m_answer)
                    imshow(m_answer);
                    imwrite(m_answer,['E:\EvaluationTool\map\my12\CVC-300\',filename(maskname).name])


                    % 函数实现，a为表达式，当a为true时，返回b，为false时，返回c
end
function  result = ternaryOperator(a,b,c)
    if a
        result = b;
    else
        result = c;
    end
end


    % subplot(2,2,3)
    % imshow(m_answer);
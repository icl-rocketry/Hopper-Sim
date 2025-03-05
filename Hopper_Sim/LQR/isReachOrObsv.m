function [reachable,observable] = isReachOrObsv(A,B,C)
% The outputs of you function should satisfy:
% reachable = true if (A,B) is reachable
% reachable = false if (A,B) is not reachable
% observable = true if (C,A) is observable
% observable = false if (C,A) is not observable
% Note that your code should return only logical (true or false) values (not 1 or 0)
% your code below

n = length(A);
Wr = B;
Wo = C;

for i = 1:n-1
    Wr = [Wr, A^i*B];
    Wo = [Wo; C*A^i];
end

if rank(Wr) == n
    reachable = true;
else
    reachable = false;
end

if rank(Wo) == n
    observable = true;
else
    observable = false;
end

end % of function - remember to leave this here

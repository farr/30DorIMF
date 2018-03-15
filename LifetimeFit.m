%Stellar MS lifetime fit, non-rotating, LMC metallicity

Mvec=10:10:500;
for(i=1:length(Mvec)),
    M=Mvec(i);
    lnT=9.604-4.196*log(M)+0.671*(log(M))^2-0.036*(log(M))^3;
    T(i)=exp(lnT);
end;
%Table B1, https://arxiv.org/pdf/1501.03794.pdf
Kohler=[60, 3.373; 100, 2.648; 150, 2.313; 200, 2.140; 300, 1.979; 500, 1.855];
%Brott, 2011 http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-source=J/A%2bA/530/A115/models
Brott=[10, 21.4; 30, 5.43; 45, 4.01; 60, 3.373];
models=[Brott; Kohler];
plot(Mvec,T,'LineWidth',3); hold on;
scatter(models(:,1), models(:,2), 'filled');hold off;
set(gca,'FontSize', 20); xlabel('Mass, M_o'), ylabel('Lifetime, Myr');


Mvec=10:10:500;
for(i=1:length(Mvec)),
    M=Mvec(i);
    lnT=9.56-4.15*log(M)+0.665*(log(M))^2-0.037*(log(M))^3;
    T(i)=exp(lnT);
end;

order=3;
polyilya=polyfit(log(models(:,1)),log(models(:,2)),order)
ilyafit=zeros(size(Mvec));
for(i=0:order),
    ilyafit=ilyafit+polyilya(i+1)*(log(Mvec)).^(order-i);
end;
plot(log(Mvec),ilyafit,'LineWidth',3); hold on;
scatter(log(models(:,1)), log(models(:,2)), 'filled'); hold off;
plot(Mvec,exp(ilyafit),'LineWidth',3); hold on;
scatter(models(:,1), models(:,2), 'filled'); hold off;
set(gca,'FontSize', 20); xlabel('Mass, M_o'), ylabel('Lifetime, Myr');

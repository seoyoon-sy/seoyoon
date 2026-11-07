%% plot_psi_comparison.m
%  선형열관류율(psi) 비교 그림 (createfigure 형식)
%   - 왼쪽 : Timestep별 psi (Model-free 적용 전 / Method ① 후 / Method ② 후)
%   - 오른쪽: 전체기간 / 필터링 적용 / Method ① / Method ② 박스플롯
%  필요: MATLAB R2020b 이상 (boxchart, exportgraphics 사용)
%  결과: psi_comparison_num.png (눈금 숫자 있음), psi_comparison_nonum.png (없음)
clear; clc; close all;

%% ===== 사용자 설정 =====================================================
% 실행하면 엑셀 파일 선택 창이 뜸 (취소하면 종료)
[fn, fp] = uigetfile({'*.xlsx;*.xlsm;*.xls','Excel 파일 (*.xlsx, *.xlsm, *.xls)'}, '엑셀 파일을 선택하세요');
if isequal(fn, 0), disp('파일 선택이 취소되었습니다.'); return; end
xlsxFile = fullfile(fp, fn);

shAll    = 'T형 모델 기본형';            % 전체기간 시트 (672개: 28일 x 24시간)
shSMW    = 'T형 모델 기본형(SMW)';       % SMW 필터링 시트 (77개)
showAllPoints = true;                    % true: 박스 4개 모두에 점 표시, false: 전체기간만
showBands     = true;                    % true: 배경 색 띠(25~75% 구간) 표시
outPrefix = 'psi_comparison';            % 저장 파일명 접두어

%% ===== 데이터 읽기 =====================================================
% 열 번호: A=1 day, B=2 time, H=8 Baseline psi, I=9 Case I psi, K=11 Case II psi
rawAll = readmatrix(xlsxFile, 'FileType','spreadsheet', 'Sheet', shAll, 'Range', 'A3:K675');
rawSMW = readmatrix(xlsxFile, 'FileType','spreadsheet', 'Sheet', shSMW, 'Range', 'A3:K675');
rawAll = rawAll(~isnan(rawAll(:,1)), :);   % 빈 행 제거
rawSMW = rawSMW(~isnan(rawSMW(:,1)), :);

psiAll    = rawAll(:, 8);    % 전체기간 Baseline           -> '전체기간' 박스
psiBefore = rawSMW(:, 8);    % SMW 구간 Baseline (적용 전)  -> '필터링 적용' 박스, 파란 원
psiM1     = rawSMW(:, 9);    % Case I  (Method ①)           -> 초록 삼각형
psiM2     = rawSMW(:, 11);   % Case II (Method ②)           -> 노란 사각형

nS = numel(psiBefore);
t  = (1:nS).';               % Timestep (필터링된 데이터의 순번)

%% ===== 그림 (숫자 있음 / 없음 두 장) ====================================
for showNum = [true false]
    createfigure(t, psiBefore, psiM1, psiM2, psiAll, showNum, showAllPoints, showBands);
    drawnow;
    if showNum, tag = 'num'; else, tag = 'nonum'; end
    outFile = sprintf('%s_%s.png', outPrefix, tag);
    exportgraphics(gcf, outFile, 'Resolution', 300);
    fprintf('저장 완료: %s  (전체기간 n=%d, SMW 필터링 n=%d)\n', outFile, numel(psiAll), nS);
end

%% ===== 로컬 함수 (스크립트 맨 아래에 위치해야 함) =======================
function createfigure(X1, Y1, Y2, Y3, Y4, showNum, showAllPoints, showBands)
%CREATEFIGURE(X1, Y1, Y2, Y3, Y4, showNum, showAllPoints, showBands)
%  X1:  Timestep 벡터
%  Y1:  Model-free 적용 전 psi (필터링 구간 Baseline)
%  Y2:  Model-free ① 적용 후 psi (Case I)
%  Y3:  Model-free ② 적용 후 psi (Case II)
%  Y4:  전체기간 Baseline psi
%  showNum: true면 y축 눈금 숫자 표시
%  showAllPoints: true면 박스플롯 4개 모두에 점 표시
%  showBands: true면 배경 색 띠 표시

yLim = [0.3 1.5];
nS = numel(X1);
Size1 = 22;
Size2 = 26;
Color1 = [0.3 0.5 0.8];     % 파랑
Color2 = [0.2 0.6 0.3];     % 초록
Color3 = [0.93 0.8 0.1];    % 노랑

% figure 생성
figure1 = figure('WindowState','maximized','Color',[1 1 1]);

% axes 생성
axes1 = axes('Parent',figure1,'Position',[0.09 0.14 0.5 0.7]);
hold(axes1,'on');
xlim(axes1,[-1 nS+1]);
ylim(axes1,yLim);

% 배경 밴드 (각 데이터의 25~75% 구간)
if showBands
    drawBand(axes1, [-1 nS+1], iqrange(Y1), Color1);
    drawBand(axes1, [-1 nS+1], iqrange(Y2), Color2);
    drawBand(axes1, [-1 nS+1], iqrange(Y3), Color3);
end

% scatter 생성
scatter(X1,Y1,Size1,Color1,'DisplayName','Model-free 적용 전','Parent',axes1,...
    'MarkerEdgeColor',[0.3 0.5 0.8],...
    'MarkerFaceAlpha',0.45,...
    'MarkerFaceColor','flat');

scatter(X1,Y2,Size2,Color2,'DisplayName','Model-free ① 적용 후','Parent',axes1,...
    'MarkerEdgeColor',[0.1 0.3 0.1],...
    'MarkerFaceAlpha',0.6,...
    'MarkerFaceColor','flat',...
    'Marker','^');

scatter(X1,Y3,Size2,Color3,'DisplayName','Model-free ② 적용 후','Parent',axes1,...
    'MarkerEdgeColor',[0.4 0.35 0],...
    'MarkerFaceAlpha',0.85,...
    'MarkerFaceColor','flat',...
    'Marker','square');

% ylabel / xlabel 생성
ylabel('\psi');
xlabel('Timestep','FontWeight','bold','FontName','Malgun Gothic');

box(axes1,'on');
hold(axes1,'off');

% 눈금 숫자 유무
if showNum
    yTickLab = {'0.3','0.6','0.9','1.2','1.5'};
else
    yTickLab = '';
end

% 나머지 axes 속성 설정
set(axes1,'FontName','Malgun Gothic','FontSize',9,'GridLineStyle','--',...
    'GridLineWidth',1,'Layer','top','LineWidth',2,'TickLength',[0.02 0.025],...
    'XGrid','on','XMinorTick','on','XTick',[0 10 20 30 40 50 60 70],...
    'XTickLabel',{'','','','','','','',''},'YGrid','on','YMinorTick','on',...
    'YTick',[0.3 0.6 0.9 1.2 1.5],'YTickLabel',yTickLab);

% legend 생성
legend1 = legend(axes1,'show');
set(legend1,...
    'Position',[0.39239852398524 0.927295247724974 0.21520295202952 0.0254095045500506],...
    'Orientation','horizontal',...
    'FontSize',8.5);

% axes 생성
axes2 = axes('Parent',figure1,'Position',[0.63 0.14 0.34 0.7]);
hold(axes2,'on');
xlim(axes2,[0.4 4.6]);
ylim(axes2,yLim);

if showBands
    drawBand(axes2, [0.4 4.6], iqrange(Y1), Color1);
    drawBand(axes2, [0.4 4.6], iqrange(Y2), Color2);
    drawBand(axes2, [0.4 4.6], iqrange(Y3), Color3);
end

% boxchart 생성 (1: 전체기간, 2: 필터링 적용, 3: Method ①, 4: Method ②)
ydata  = {Y4, Y1, Y2, Y3};
faceC  = {[0.95 0.6 0.4], [0.685 0.775 0.91], [0.56 0.78 0.615], [0.9615 0.89 0.505]};
edgeC  = {[0.55 0.25 0.15], [0.21 0.35 0.56], [0.14 0.42 0.21], [0.55 0.45 0]};
pointC = {[0.9 0.3 0.2], [0.3 0.5 0.8], [0.2 0.6 0.3], [0.85 0.7 0]};

rng(1);   % 점 흩뿌림(지터) 재현용 시드
for k = 1:4
    v = ydata{k};
    boxchart(k*ones(size(v)),v,'Parent',axes2,'MarkerStyle','none',...
        'LineWidth',0.8,...
        'WhiskerLineColor',edgeC{k},...
        'BoxFaceAlpha',0.75,...
        'BoxMedianLineColor',edgeC{k},...
        'BoxEdgeColor',edgeC{k},...
        'BoxFaceColor',faceC{k},...
        'BoxWidth',0.6);
    % 점 표시 (showAllPoints=false면 전체기간만)
    if k == 1 || showAllPoints
        jit = (rand(size(v)) - 0.5) * 0.5;
        scatter(k + jit, v, 6, pointC{k}, 'filled', 'Parent', axes2, ...
            'MarkerFaceAlpha', 0.5, 'HandleVisibility', 'off');
    end
end

box(axes2,'on');
hold(axes2,'off');
% 나머지 axes 속성 설정
set(axes2,'FontName','Malgun Gothic','FontSize',9,'Layer','top','LineWidth',...
    0.6,'XTick',[1 2 3 4],'XTickLabel',...
    {'전체\newline기간','필터링\newline적용','Method\newline①','Method\newline②'},...
    'YTick',[0.3 0.6 0.9 1.2 1.5],'YTickLabel','');
end

function r = iqrange(v)
% 25%, 75% 분위수 (Statistics Toolbox 없이 계산)
    v = sort(v(:));
    n = numel(v);
    r = interp1(((1:n)-0.5)/n, v, [0.25 0.75], 'linear', 'extrap');
end

function drawBand(ax, xl, y, c)
    patch(ax, [xl(1) xl(2) xl(2) xl(1)], [y(1) y(1) y(2) y(2)], c, ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    line(ax, xl, [y(1) y(1)], 'Color', c*0.8, 'LineWidth', 0.7, 'HandleVisibility', 'off');
    line(ax, xl, [y(2) y(2)], 'Color', c*0.8, 'LineWidth', 0.7, 'HandleVisibility', 'off');
end

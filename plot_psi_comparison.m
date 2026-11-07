%% plot_psi_comparison.m
%  선형열관류율(psi) 비교 그림
%   - 왼쪽 : Timestep별 psi (Model-free 적용 전 / Method ① 후 / Method ② 후)
%   - 오른쪽: 전체기간 / 필터링 적용 / Method ① / Method ② 박스플롯
%  필요: MATLAB R2020b 이상 (boxchart, exportgraphics 사용)
clear; clc; close all;

%% ===== 사용자 설정 =====================================================
xlsxFile = '엑셀파일명.xlsx';            % 엑셀 파일 경로 (예: 'C:\data\result.xlsx')
shAll    = 'T형 모델 기본형';            % 전체기간 시트 (672개: 28일 x 24시간)
shSMW    = 'T형 모델 기본형(SMW)';       % SMW 필터링 시트 (77개)
fontName = 'Malgun Gothic';              % Mac이면 'AppleGothic'
yLim     = [0.3 1.5];
showAllPoints = false;                   % true: 4개 박스 모두에 점 표시 (false: 전체기간만)
outFile  = 'psi_comparison.png';

%% ===== 데이터 읽기 =====================================================
% 열 번호: A=1 day, B=2 time, H=8 Baseline psi, I=9 Case I psi, K=11 Case II psi
rawAll = readmatrix(xlsxFile, 'Sheet', shAll, 'Range', 'A3:K675');
rawSMW = readmatrix(xlsxFile, 'Sheet', shSMW, 'Range', 'A3:K675');
rawAll = rawAll(~isnan(rawAll(:,1)), :);   % 빈 행 제거
rawSMW = rawSMW(~isnan(rawSMW(:,1)), :);

psiAll    = rawAll(:, 8);    % 전체기간 Baseline          -> '전체기간' 박스
psiBefore = rawSMW(:, 8);    % SMW 구간 Baseline (적용 전) -> '필터링 적용' 박스, 파란 원
psiM1     = rawSMW(:, 9);    % Case I  (Method ①)          -> 초록 삼각형
psiM2     = rawSMW(:, 11);   % Case II (Method ②)          -> 노란 사각형

nS = numel(psiBefore);
t  = (1:nS).';               % Timestep (필터링된 데이터의 순번)

%% ===== 색상 ===========================================================
cBlue   = [0.30 0.50 0.80];
cGreen  = [0.20 0.60 0.30];
cYellow = [0.93 0.80 0.10];
cRed    = [0.90 0.30 0.20];
lighten = @(c,a) c + (1-c)*a;

%% ===== Figure / Axes ==================================================
fig = figure('Color','w','Units','centimeters','Position',[5 5 15 9.5]);
ax1 = axes(fig, 'Position', [0.09 0.14 0.50 0.70]);   % 왼쪽: Timestep
ax2 = axes(fig, 'Position', [0.63 0.14 0.34 0.70]);   % 오른쪽: 박스플롯
hold(ax1,'on'); hold(ax2,'on');

xl1 = [0 nS+1];
xl2 = [0.4 4.6];

%% ===== 배경 밴드 (각 데이터의 25~75% 구간) ============================
bandB  = iqrange(psiBefore);
bandM1 = iqrange(psiM1);
bandM2 = iqrange(psiM2);
for ax = [ax1 ax2]
    xl = xlim_of(ax, xl1, xl2, ax1);
    drawBand(ax, xl, bandB,  cBlue);
    drawBand(ax, xl, bandM1, cGreen);
    drawBand(ax, xl, bandM2, cYellow);
end

%% ===== 왼쪽: Timestep 그래프 ===========================================
plot(ax1, t, psiBefore, '-', 'Color', lighten(cBlue,0.5),   'LineWidth', 0.6, 'HandleVisibility','off');
plot(ax1, t, psiM1,     '-', 'Color', lighten(cGreen,0.5),  'LineWidth', 0.6, 'HandleVisibility','off');
plot(ax1, t, psiM2,     '-', 'Color', lighten(cYellow,0.3), 'LineWidth', 0.6, 'HandleVisibility','off');

h1 = scatter(ax1, t, psiBefore, 22, cBlue,   'o', 'filled', 'MarkerFaceAlpha',0.45, 'MarkerEdgeColor',cBlue);
h2 = scatter(ax1, t, psiM1,     26, cGreen,  '^', 'filled', 'MarkerFaceAlpha',0.60, 'MarkerEdgeColor',[0.1 0.3 0.1]);
h3 = scatter(ax1, t, psiM2,     26, cYellow, 's', 'filled', 'MarkerFaceAlpha',0.85, 'MarkerEdgeColor',[0.4 0.35 0]);

xlim(ax1, xl1); ylim(ax1, yLim);
xlabel(ax1, 'Timestep', 'FontName', fontName, 'FontSize', 10, 'FontWeight','bold');
ylabel(ax1, '\psi', 'Interpreter','tex', 'FontSize', 13);
ax1.XTick = [];
ax1.YTick = yLim(1):0.3:yLim(2);

%% ===== 오른쪽: 박스플롯 ================================================
data  = {psiAll, psiBefore, psiM1, psiM2};
fcol  = {[0.95 0.60 0.40], lighten(cBlue,0.55), lighten(cGreen,0.45), lighten(cYellow,0.45)};
ecol  = {[0.55 0.25 0.15], cBlue*0.7, cGreen*0.7, [0.55 0.45 0]};
ptcol = {cRed, cBlue, cGreen, [0.85 0.70 0]};

rng(1);                                   % 지터 재현용 시드
for k = 1:4
    v = data{k};
    boxchart(ax2, k*ones(size(v)), v, ...
        'BoxFaceColor', fcol{k}, 'BoxFaceAlpha', 0.75, ...
        'BoxEdgeColor', ecol{k}, 'WhiskerLineColor', ecol{k}, ...
        'BoxMedianLineColor', ecol{k}, 'LineWidth', 0.8, ...
        'MarkerStyle', 'none', 'BoxWidth', 0.6);
    if k == 1 || showAllPoints
        jit = (rand(size(v)) - 0.5) * 0.5;
        scatter(ax2, k + jit, v, 5, ptcol{k}, 'filled', ...
            'MarkerFaceAlpha', 0.45, 'HandleVisibility','off');
    end
end

xlim(ax2, xl2); ylim(ax2, yLim);
ax2.XTick = 1:4;
ax2.XTickLabel = {'전체\newline기간', '필터링\newline적용', 'Method\newline①', 'Method\newline②'};
ax2.YTick = ax1.YTick;
ax2.YTickLabel = [];

%% ===== 공통 서식 ======================================================
for ax = [ax1 ax2]
    ax.Box = 'on';
    ax.FontName = fontName;
    ax.FontSize = 9;
    ax.LineWidth = 0.6;
    ax.TickDir = 'in';
    ax.Layer = 'top';
end
ax1.XLabel.FontName = fontName;

%% ===== 범례 (그림 상단 중앙) ===========================================
lgd = legend(ax1, [h1 h2 h3], ...
    {'Model-free 적용 전', 'Model-free ① 적용 후', 'Model-free ② 적용 후'}, ...
    'Orientation','horizontal', 'FontName',fontName, 'FontSize',8.5, 'Box','on');
lgd.Units = 'normalized';
lgd.Position = [0.14 0.905 0.72 0.07];

%% ===== 저장 ===========================================================
exportgraphics(fig, outFile, 'Resolution', 300);
fprintf('저장 완료: %s  (전체기간 n=%d, SMW 필터링 n=%d)\n', outFile, numel(psiAll), nS);

%% ===== 로컬 함수 (스크립트 맨 아래에 위치해야 함) =======================
function r = iqrange(v)
% 25%, 75% 분위수 (Statistics Toolbox 없이 계산)
    v = sort(v(:));
    n = numel(v);
    r = interp1(((1:n)-0.5)/n, v, [0.25 0.75], 'linear', 'extrap');
end

function xl = xlim_of(ax, xl1, xl2, ax1)
    if ax == ax1, xl = xl1; else, xl = xl2; end
end

function drawBand(ax, xl, y, c)
    patch(ax, [xl(1) xl(2) xl(2) xl(1)], [y(1) y(1) y(2) y(2)], c, ...
        'FaceAlpha', 0.18, 'EdgeColor', 'none', 'HandleVisibility', 'off');
    line(ax, xl, [y(1) y(1)], 'Color', c*0.8, 'LineWidth', 0.7, 'HandleVisibility', 'off');
    line(ax, xl, [y(2) y(2)], 'Color', c*0.8, 'LineWidth', 0.7, 'HandleVisibility', 'off');
end

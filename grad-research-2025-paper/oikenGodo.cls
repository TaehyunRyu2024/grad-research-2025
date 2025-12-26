%%%
%% skysk.cls (pLaTeX2e)
%% Shinya Fujie
%% E-mail: fujie@pcl.cs.waseda.ac.jp
%%%
%% oikenGodo.cls (modified on 2014/12/09 by Kohei Yatabe)
%%
\NeedsTeXFormat{pLaTeX2e}
\ProvidesClass{oikenGodo}
[2004/01/09 v1.0 for Shirai Kobayashi Yamasaki Sagisaka Kikuchi Lab. Meeting]

\newif\if@usejistfm \@usejistfmfalse
\DeclareOption{usejistfm}{\@usejistfmtrue}

%% for 2.1.4 and non pLaTeX2e compatibility
\@ifundefined{hour}{}{%
 \hour\time \divide\hour by 60\relax
 \@tempcnta\hour \multiply\@tempcnta 60\relax
 \minute\time \advance\minute-\@tempcnta
}

\newif\if@NTT
\@ifundefined{kanjiskip}{\@NTTtrue
 \let\kanjiskip\jintercharskip
 \let\xkanjiskip\jasciikanjiskip
 }{\@NTTfalse\let\jmathkanjiskip\xkanjiskip}
\@ifundefined{iftdir}{\newif\iftdir\tdirfalse}{}
\@ifundefined{burasage}{}{\burasage`，=1\burasage`．=1}

\newdimen\@JQ
 \@JQ=0.7392507pt
\newdimen\@Q
 \@Q=0.71144pt
\newdimen\@EM
 \@EM=.03125em

\if@NTT\else
 \newif\if@enablejfam \@enablejfamtrue
\fi
\if@NTT\else
 \DeclareOption{disablejfam}{\@enablejfamfalse}
\fi
\ProcessOptions\relax
% a4paper
\setlength\paperheight {297mm}
\setlength\paperwidth  {210mm}
% twoside
\@twosidetrue
% twocolumn
\@twocolumntrue
% final
\setlength\overfullrule{0pt}
% 11pt
\newcommand{\@ptsize}{1}

\if@NTT\else
 \if@usejistfm
 \DeclareFontShape{JY1}{mc}{m}{n}{%
     <5> <6> <7> <8> <9> <10> %sgen*min
     <10.95><12><14.4><17.28><20.74><24.88> jis
     <-> jis
     }{}
 \DeclareFontShape{JY1}{mc}{bx}{n}{<->ssub*gt/m/n}{}
 \DeclareFontShape{JY1}{gt}{m}{n}{%
     <5> <6> <7> <8> <9> <10> %sgen*goth
     <10.95><12><14.4><17.28><20.74><24.88> jisg
     <-> jisg
     }{}
 \DeclareFontShape{JY1}{gt}{bx}{n}{<->ssub*gt/m/n}{}
 \fi
%% JY1 it
\DeclareFontShape{JY1}{mc}{m}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JY1}{mc}{bx}{it}{<->ssub*gt/m/n}{}
\DeclareFontShape{JY1}{gt}{m}{it}{<->ssub*gt/m/n}{}
\DeclareFontShape{JY1}{gt}{bx}{it}{<->ssub*gt/m/n}{}
%%
\DeclareFontShape{JY1}{mc}{b}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JY1}{mc}{eb}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JY1}{gt}{b}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JY1}{gt}{eb}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JY1}{gt}{ub}{it}{<->ssub*mc/m/n}{}
%% JT1 it
\DeclareFontShape{JT1}{mc}{m}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JT1}{mc}{bx}{it}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{gt}{m}{it}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{gt}{bx}{it}{<->ssub*gt/m/n}{}
%%
\DeclareFontShape{JT1}{mc}{b}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JT1}{mc}{eb}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JT1}{gt}{b}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JT1}{gt}{eb}{it}{<->ssub*mc/m/n}{}
\DeclareFontShape{JT1}{gt}{ub}{it}{<->ssub*mc/m/n}{}
%% sc
\DeclareFontShape{JY1}{mc}{m}{sc}{<->ssub*mc/m/n}{}
\DeclareFontShape{JT1}{mc}{m}{sc}{<->ssub*mc/m/n}{}
%% n
\DeclareFontShape{JY1}{gt}{b}{n}{<->ssub*gt/m/n}{}
\DeclareFontShape{JT1}{gt}{b}{n}{<->ssub*gt/m/n}{}
\fi

%% from /usr/local/teTeX/share/texmf/tex/latex/base/ot1cmr.fd
\DeclareFontShape{OT1}{cmr}{m}{n}%
     {%
      <-6>cmr5%
      <6-7>cmr6%
      <7-8>cmr7%
      <8-9>cmr8%
      <9-10>cmr9%
      <10-12>cmr10%
      <12-17>cmr12%
      <17->cmr17%
      }{}
\DeclareFontShape{OT1}{cmr}{m}{sl}%
     {%
      <-9>cmsl8%
      <9-10>cmsl9%
      <10-12>cmsl10%
      <12->cmsl12%
      }{}
\DeclareFontShape{OT1}{cmr}{m}{it}%
     {%
      <-8>cmti7%
      <8-9>cmti8%
      <9-10>cmti9%
      <10-12>cmti10%
      <12->cmti12%
      }{}
\DeclareFontShape{OT1}{cmr}{m}{sc}%
     {%
      <->cmcsc10%
      }{}
\DeclareFontShape{OT1}{cmr}{m}{ui}
     {%
      <->cmu10%
      }{}
\DeclareFontShape{OT1}{cmr}{b}{n}
     {%
      <->cmb10%
      }{}
\DeclareFontShape{OT1}{cmr}{bx}{n}
     {%
      <-6>cmbx5%
      <6-7>cmbx6%
      <7-8>cmbx7%
      <8-9>cmbx8%
      <9-10>cmbx9%
      <10-12>cmbx10%
      <12->cmbx12%
      }{}
\DeclareFontShape{OT1}{cmr}{bx}{sl}
     {%
      <->cmbxsl10%
      }{}
\DeclareFontShape{OT1}{cmr}{bx}{it}
     {%
      <->cmbxti10%
      }{}
\DeclareFontShape{OT1}{cmr}{bx}{ui}
     {%
      <->sub*cmr/m/ui
      }{}

 \renewcommand{\normalsize}{%
  \@setfontsize\normalsize\@ixpt{20.5\@Q}%
  \kanjiskip\z@ \@plus .1zw \@minus .05zw
  \abovedisplayskip 9\p@ \@plus 3\p@ \@minus 4\p@
  \abovedisplayshortskip \z@ \@plus 2.5\p@
  \belowdisplayshortskip 5\p@ \@plus 2\p@ \@minus 2\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}
 \normalsize
 \setbox0\hbox{\char\euc"A1A1}% "
 \setlength\Cht{\ht0}
 \setlength\Cdp{\dp0}
 \setlength\Cwd{\wd0}
 \setlength\Cvs{\baselineskip}
 \setlength\Chs{\wd0}

 \newcommand{\small}{%
  \@setfontsize\small\@viiipt{18\@Q}%
  \kanjiskip\z@ \@plus .1zw \@minus .05zw
  \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
  \abovedisplayshortskip \z@ \@plus\p@
  \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
  \def\@listi{\leftmargin\leftmargini
              \topsep 3\p@ \@plus \p@ \@minus \p@
              \parsep 1\p@ \@plus \p@ \@minus \p@
              \itemsep \parsep}%
  \belowdisplayskip \abovedisplayskip}

 \newcommand{\footnotesize}{%
  \@setfontsize\footnotesize\@viipt{16\@Q}%
  \kanjiskip\z@ \@plus .1zw \@minus .05zw
  \abovedisplayskip 5\p@ \@plus 1.5\p@ \@minus 3\p@
  \abovedisplayshortskip \z@ \@plus\p@
  \belowdisplayshortskip 2.5\p@ \@plus \p@ \@minus 1.5\p@
  \def\@listi{\leftmargin\leftmargini
              \topsep 2\p@ \@plus\p@ \@minus\p@
              \parsep 1\p@ \@plus\p@ \@minus\p@
              \itemsep \parsep}%
  \belowdisplayskip \abovedisplayskip}

 \newcommand{\scriptsize}{\@setfontsize\scriptsize\@vipt\@viiipt}
 \newcommand{\tiny}{\@setfontsize\tiny\@vpt\@vipt}
 \newcommand{\large}{\@setfontsize\large\@xpt{22\@Q}}
 \newcommand{\Large}{\@setfontsize\Large\@xipt{27\@Q}}
 \newcommand{\LARGE}{\@setfontsize\LARGE\@xiipt{33\@Q}}
 \newcommand{\huge}{\@setfontsize\huge\@xivpt{25}}
 \newcommand{\Huge}{\@setfontsize\Huge\@xviipt{30}}

 %\headheight \z@
 \headheight 0mm
 %\headsep    \z@
 \headsep 0mm
 \topskip    1\Cht
 %\footskip   9.5mm
 \footskip 50.0mm
 \maxdepth .5\topskip
 %\textwidth 58\Cwd
 \textwidth 159.2mm
 \@settopoint\textwidth
 %\textheight 49\Cvs
 \textheight 246.2mm
 % \advance \textheight \topskip
 % \advance \textheight .15mm
 \@settopoint\textheight

\if@twocolumn
  \setlength\marginparsep{5mm}
\else
  \setlength\marginparsep{5mm}
\fi
\setlength\marginparpush{5pt}

\iftombow
% \hoffset\z@
% \voffset\z@
 \topmargin\paperheight
 \advance \topmargin -\textheight
 \advance \topmargin -\headheight
 \advance \topmargin -\headsep
 \advance \topmargin -\footskip
 \divide \topmargin \tw@
 \advance \topmargin -1.25mm
 \advance \topmargin -1in
 \@settopoint\topmargin
 \oddsidemargin\paperwidth
 \advance \oddsidemargin -\textwidth
 \divide \oddsidemargin \tw@
 \advance \oddsidemargin -1in
 \@settopoint\oddsidemargin
 \evensidemargin\oddsidemargin
 \@settopoint\evensidemargin
\else
 \topmargin \z@
 \@settopoint\topmargin
 \oddsidemargin\z@
 \@settopoint\oddsidemargin
 \evensidemargin\z@
 \@settopoint\evensidemargin
\fi

\marginparwidth 20mm
\@settopoint\marginparwidth

\setlength\footnotesep{2.8mm}
\setlength{\skip\footins}{1\Cvs plus .9\Cvs \@minus .3\Cvs}
\setlength\intextsep   {1\Cvs \@plus .75\Cvs \@minus .25\Cvs}
\setlength\floatsep    {1\Cvs \@plus .75\Cvs \@minus .25\Cvs}
\setlength\dblfloatsep {1\Cvs \@plus .75\Cvs \@minus .25\Cvs}
\setlength\textfloatsep{1.5\Cvs \@plus .75\Cvs \@minus .25\Cvs}
\setlength\dbltextfloatsep{1.5\Cvs \@plus .75\Cvs \@minus .25\Cvs}

\setlength\@fptop{0\p@ \@plus 1fil}
\setlength\@fpsep{8\p@ \@plus 2fil}
\setlength\@fpbot{0\p@ \@plus 1fil}
\setlength\@dblfptop{0\p@ \@plus 1fil}
\setlength\@dblfpsep{8\p@ \@plus 2fil}
\setlength\@dblfpbot{0\p@ \@plus 1fil}
\setlength\partopsep{2\p@ \@plus 1\p@ \@minus 1\p@}

\def\@listi{\leftmargin\leftmargini
  \parsep  4\p@ \@plus2\p@ \@minus\p@
  \topsep  8\p@ \@plus2\p@ \@minus4\p@
  \itemsep 4\p@ \@plus2\p@ \@minus\p@}
\let\@listI\@listi
\@listi
\def\@listii{\leftmargin\leftmarginii
   \labelwidth\leftmarginii \advance\labelwidth-\labelsep
   \topsep\z@ \@plus\p@ \@minus\p@
   \parsep\z@ \@plus\p@ \@minus\p@
   \itemsep\parsep}
\def\@listiii{\leftmargin\leftmarginiii
   \labelwidth\leftmarginiii \advance\labelwidth-\labelsep
   \topsep\z@ \@plus\p@ \@minus\p@
   \parsep\z@
   \partopsep\z@ \@plus\z@ \@minus\p@
   \itemsep\topsep}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}

\columnsep 2zw
\columnseprule 0\p@
\lineskip 2\p@ 
\normallineskip 2\p@
\renewcommand{\baselinestretch}{}
\parskip\z@    %% 0\p@ \@plus \p@
\parindent 1zw %% 1\Cwd

\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{6}
\setcounter{bottomnumber}{5}
\setcounter{totalnumber}{11}
\setcounter{dbltopnumber}{5}
\renewcommand{\topfraction}{.95}
\renewcommand{\bottomfraction}{.95}
\renewcommand{\textfraction}{.05}
\renewcommand{\floatpagefraction}{.05}
\renewcommand{\dbltopfraction}{.95}
\renewcommand{\dblfloatpagefraction}{.05}

\let\@Rtop\relax\let\@Ltop\relax\let\@Rbot\relax\let\@Lbot\relax

 \def\ps@plain{\let\@mkboth\@gobbletwo
   \let\ps@jpl@in\ps@plain
   \def\@oddhead{\@Ltop\hfil\@Rtop}%
   \def\@oddfoot{\@Lbot\reset@font\hfil ---\ \thepage\ ---\@Rbot}% \hfil
   \let\@evenhead\@oddhead
   \let\@evenfoot\@oddfoot}

%%% 研究室名
\def\labname#1{\gdef\@labname{#1}}
\def\@labname{}
%%% 班
\def\group#1{\gdef\@group{#1}}
\def\@group{}
%%% 学年
\def\grade#1#2{\gdef\@master{#1}\gdef\@grade{#2}}
\def\@master{}
\def\@grade{}

\def\titlepagebaselinestretch#1{\gdef\@titlepagebaselinestretch{#1}}
\titlepagebaselinestretch{1}
\def\maketitle{\par
    \twocolumn[\@maketitle]%
}

\def\@maketitle{%
  \setbox\@tempboxa\vbox{%
    \def\baselinestretch{\@titlepagebaselinestretch}%
    \bgroup
     \centering
     {\fontsize{16}{23}\selectfont
      \@title\par
     }%
     \vskip.25\baselineskip
     {\LARGE
	\@labname\hskip1zw
	\ifx\@group\empty\else\@group\hskip1zw\fi
	\if\@master m 修士\fi%
	\if\@grade\empty\else\@grade 年\hskip1zw\fi
	\@author\par
     }%
     \vskip.5\baselineskip
     \egroup
     \vskip\baselineskip
  }%
  \@tempdima\ht\@tempboxa
  \advance \@tempdima \dp\@tempboxa %% <
  \ifdim \@tempdima>\textheight
   \ieicej@warning{The title page is too high. 
      ^^J\space\space You can adjust the height of a title page
      ^^J\space\space by changing \string\titlepagebaselinestretch\space 
		      in preamble.
      ^^J\space\space For example, \string\titlepagebaselinestretch{0.9}}%
   {\parindent\z@
    \vtop to \z@{%
     \vbox to \textheight{%
      \unvbox\@tempboxa
     }\vss
    }%
   }\par
   \clearpage
  \else
   \divide \@tempdima \Cvs
   \@tempcnta=\@tempdima
   \advance \@tempcnta \@ne
   \vbox to \@tempcnta\Cvs{%
    \box\@tempboxa\vss
   }%
  \fi
 }

\setcounter{secnumdepth}{5}
\newcounter{section}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]
\newcounter{subparagraph}[paragraph]
\renewcommand{\thesection}{\@arabic\c@section.}
\renewcommand{\thesubsection}{\thesection\,\@arabic\c@subsection}
\renewcommand{\thesubsubsection}{%
   \thesubsection.\,\@arabic\c@subsubsection}
\renewcommand{\theparagraph}{% (
 \@alph\c@paragraph\,)}
\renewcommand{\thesubparagraph}{% (
 \@roman\c@subparagraph\,)}

\def\@afterheading{%
 \global\@nobreaktrue
 \everypar{%
  \if@nobreak
   \global\@nobreakfalse
   \clubpenalty \z@
    \if@afterindent
    \else
     {\setbox\z@=\lastbox}%
    \fi
  \else
   \clubpenalty\@clubpenalty
   \everypar{}%
  \fi}%
}

\def\@startsection#1#2#3#4#5#6{\if@noskipsec \leavevmode \fi
   \par \@tempskipa #4\relax
   \@afterindenttrue
   \ifdim \@tempskipa <\z@ \@tempskipa -\@tempskipa \@afterindentfalse\fi %% >
   \if@nobreak
    \everypar{}%
   \else
    \addpenalty\@secpenalty
     \ifnum #2=\@ne
       \vskip-\prevdepth \prevdepth\z@ \vskip\Cvs
       \advance\@tempskipa-\Cvs \vspace*{\@tempskipa}%
     \else
      \addvspace\@tempskipa
     \fi
   \fi
 \@ifstar{\@ssect{#3}{#4}{#5}{#6}}{\@dblarg{\@sect{#1}{#2}{#3}{#4}{#5}{#6}}}}

%% \@sect{NAME}{LEVEL}{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}[ARG1]{ARG2}
\def\@sect#1#2#3#4#5#6[#7]#8{%% <
 \ifnum #2>\c@secnumdepth
  \let\@svsec\@empty
 \else
  \refstepcounter{#1}%
  \protected@edef\@svsec{\@seccntformat{#1}\relax}%
 \fi
  \@tempskipa #5\relax %% <
  \ifdim \@tempskipa>\z@
   \begingroup
    #6{%
      \@hangfrom{\hskip #3\relax{\csname #2@font\endcsname \@svsec}}%
       \interlinepenalty \@M \@secjidori{#8}\@@par}%
   \endgroup
  \else
   \def\@svsechd{%
     #6{\hskip #3\relax
     \@svsec #8}}%
  \fi
 \@xsect{#5}}

%% \@ssect{INDENT}{BEFORESKIP}{AFTERSKIP}{STYLE}{ARG} ==
\def\@ssect#1#2#3#4#5{%
  \@tempskipa #3\relax %% <
  \ifdim \@tempskipa>\z@
    \begingroup
      #4{%
        \@hangfrom{\hskip #1}%
          \interlinepenalty \@M \@secjidori{#5}\@@par}%
    \endgroup
  \else
    \def\@svsechd{#4{\hskip #1\relax #5}}%
  \fi
  \@xsect{#3}}

\def\@secjidori#1{%
 \def\@tempfn{}%
 \def\@tempfncnt{}%
 \bgroup
  \def\footnote{\@ifnextchar[% ]
   {\@ignorefn}{\@ignorefn[]}}%
  \long\def\@ignorefn[##1]##2{\gdef\@tempfncnt{##1}\gdef\@tempfn{##2}}%
  \global\setbox\@tempboxb\hbox{\inhibitglue #1}%
 \egroup %% <
  \ifdim \wd\@tempboxb>4.5zw
   \inhibitglue #1%
  \else
   \def\@makefnmark{\kern0pt\hbox to \z@{%
    $^{\mbox{\tiny \kern-.5zw （注\@thefnmark{}）\kern-.5zw}}$\hss}}%
   \kanjiskip\z@ \@plus 1fill \@minus 1fill\xkanjiskip=\kanjiskip
    \jmathkanjiskip=\kanjiskip
   \hbox to 5zw{\inhibitglue #1\hfil}%
   \ifx\@tempfn\empty\else
    \ifx\@tempfncnt\empty\footnotetext{\@tempfn}\else
     \footnotetext[\@tempfncnt]{\@tempfn}
    \fi\fi
  \fi
}

\@namedef{3@font}{\bfseries}
\def\@seccntformat#1{\csname the#1\endcsname\hskip1zw}

\def\sec@font{\bfseries\mathversion{bold}\baselineskip20.5\@Q}

 \newcommand{\section}{\@startsection{section}{1}{1zw}%
   {.5\Cvs \@plus.1mm \@minus.1mm}%
   {.5\Cvs \@plus.1mm \@minus.1mm}%
   {\reset@font\Large\sec@font}}
 \newcommand{\subsection}{\@startsection{subsection}{2}{1zw}%
   {.00001\Cvs \@plus.1mm \@minus.1mm}%
   {.00001\Cvs \@plus.1mm \@minus.1mm}%
   {\reset@font\normalsize\bfseries\mathversion{bold}}}
 \newcommand{\subsubsection}{\@startsection{subsubsection}{3}{1zw}%
   {.00001\Cvs \@plus.1mm \@minus.1mm}%
   {.00001\Cvs \@plus.1mm \@minus.1mm}%
   {\reset@font\normalsize}}
\newcommand{\paragraph}{\@startsection{paragraph}{4}{1zw}%
   {.00001\Cvs \@plus.1mm \@minus.1mm}%
   {.00001\Cvs \@plus.1mm \@minus.1mm}%
   {\reset@font\normalsize}}
\newcommand{\subparagraph}{\@startsection{subparagraph}{5}{1zw}%
   {.00001\Cvs \@plus.1mm \@minus.1mm}%
   {.00001\Cvs \@plus.1mm \@minus.1mm}%
   {\reset@font\normalsize}}

\newcommand{\appendix}{\par
 \renewcommand{\section}{\@startsection{section}{1}{\z@}
  {.5\Cvs \@plus.1mm \@minus.1mm}
  {.5\Cvs \@plus.1mm \@minus.1mm}
  {%\if@letter\centering\bfseries\else
  \centering\normalsize\sec@font
  %\fi
  }}%
 \section*{付録}
  \renewcommand{\section}{\@startsection{section}{1}{1zw}
   {.00001\Cvs \@plus.1mm \@minus.1mm}
   {.00001\Cvs \@plus.1mm \@minus.1mm}
   {\normalsize\bfseries\mathversion{bold}}}%
 \setcounter{section}{0}%
 \setcounter{subsection}{0}%
 \setcounter{equation}{0}%
 \setcounter{figure}{0}%
 \setcounter{table}{0}%
 \def\theequation{\protect\mbox{A$\cdot$\@arabic\c@equation}}%
 \def\thefigure{A$\cdot$\,\@arabic\c@figure}%
 \def\thetable{A$\cdot$\,\@arabic\c@table}%
}

\if@twocolumn
  \setlength\leftmargini {2zw}%
\else
  \setlength\leftmargini {2zw}%
\fi
\setlength\leftmarginii  {2zw}%
\setlength\leftmarginiii {2zw}%
\setlength\leftmarginiv  {2zw}%
\if@twocolumn
  \setlength\leftmarginv {1zw}%
  \setlength\leftmarginvi{1zw}%
\else
  \setlength\leftmarginv {1zw}%
  \setlength\leftmarginvi{1zw}%
\fi
\setlength  \labelsep  {1zw}%
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty

\renewcommand{\theenumi}{\@arabic\c@enumi}
\renewcommand{\theenumii}{\@alph\c@enumii}
\renewcommand{\theenumiii}{\@roman\c@enumiii}
\renewcommand{\theenumiv}{\@Alph\c@enumiv}
\newcommand{\labelenumi}{\kern-.5zw （\,\theenumi\,）\kern-.5zw}
\newcommand{\labelenumii}{\kern-.5zw （\,\theenumii\,）\kern-.5zw}
\newcommand{\labelenumiii}{\theenumiii.}
\newcommand{\labelenumiv}{\theenumiv.}
\renewcommand{\p@enumii}{\theenumi}
\renewcommand{\p@enumiii}{\theenumi(\theenumii)}
\renewcommand{\p@enumiv}{\p@enumiii\theenumiii}

\renewenvironment{enumerate}
  {%% <
   \ifnum \@enumdepth >3\relax\@toodeep\else
    \advance\@enumdepth\@ne
    \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
    \list{\csname label\@enumctr\endcsname}{%
     \leftmargin\z@
     \labelwidth2zw
     \labelsep1zw
     \itemindent4zw
     \listparindent1zw
     \topsep\z@\parsep\z@\partopsep\z@\itemsep\z@
     \clubpenalty-200
     \usecounter{\@enumctr}%
     \def\makelabel##1{\hss\llap{##1}}}%
    \fi}{\endlist}

\newcommand{\labelitemi}{\leavevmode
 \raise.1zh\hbox to 1zw{\hss \textbullet\hss}}
\newcommand{\labelitemii}{{\normalfont\bfseries\textendash}}
\newcommand{\labelitemiii}{\textasteriskcentered}
\newcommand{\labelitemiv}{\textperiodcentered}

\renewenvironment{itemize}
  {%% <
   \ifnum \@itemdepth >3\relax\@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
%%  \expandafter
    \list{\csname \@itemitem\endcsname}{%
     \topsep\z@\parsep\z@\partopsep\z@\itemsep\z@
     \leftmargin\z@
     \labelwidth1zw
     \labelsep1zw
     \itemindent3zw
     \listparindent1zw
     \clubpenalty-200\relax
     \def\makelabel##1{\hss\llap{##1}}}%
    \fi}{\endlist}

\def\descriptionlabel#1{\hspace\labelsep #1}
\newenvironment{description}{%
 \list{}{%
  \leftmargin\z@
  \labelsep1zw
  \labelwidth\z@
  \itemindent-\leftmargin
  \listparindent1zw
  \topsep\z@\parsep\z@\partopsep\z@\itemsep\z@
  \let\makelabel\descriptionlabel}}{\endlist}

\newskip\@hlskip
\@hlskip.5\Cvs \@plus 1mm \@minus .5mm

\newenvironment{verse}
  {\let\\\@centercr
   \list{}{\itemsep\z@
           \topsep\@hlskip\parsep\z@\partopsep\z@
           \itemindent -1zw
           \listparindent\itemindent
           \rightmargin\leftmargin \advance\leftmargin 1zw}%
           \item\relax}{\endlist}

\newenvironment{quotation}
  {\list{}{\listparindent 1zw
           \itemindent\listparindent
           \rightmargin\leftmargin
           \topsep\@hlskip\parsep\z@\partopsep\z@
           }%
           \item\relax}{\endlist}

\newenvironment{quote}
  {\list{}{%
           \topsep\@hlskip\parsep\z@\partopsep\z@
           \rightmargin\leftmargin}%
           \item\relax}{\endlist}

\gdef\tabbing{\lineskip \z@skip\let\>\@rtab\let\<\@ltab\let\=\@settab
     \let\+\@tabplus\let\-\@tabminus\let\`\@tabrj\let\'\@tablab
     \let\\=\@tabcr
     \@hightab\@firsttab
     \global\@nxttabmar\@firsttab
     \dimen\@firsttab\@totalleftmargin
     \global\@tabpush\z@ \global\@rjfieldfalse
     \topsep\@hlskip \parsep\z@\partopsep\z@
     \trivlist \item\relax
     \if@minipage\else\vskip\parskip\fi
     \setbox\@tabfbox\hbox{%
       \rlap{\hskip\@totalleftmargin\indent\the\everypar}}%
     \def\@itemfudge{\box\@tabfbox}%
     \@startline\ignorespaces}

\newdimen\verbatimleftmargin
  \verbatimleftmargin\z@
\newdimen\verbatimbaselineskip
  \verbatimbaselineskip\Cvs
\def\verbatimsize{\normalsize}

\def\@verbatim{%
 \topsep\z@\partopsep\z@\parsep\z@\itemsep\z@
 \trivlist \item\relax
  \if@minipage\else
   \vskip\baselineskip
   \vskip-\verbatimbaselineskip
  \fi
  \leftskip\@totalleftmargin
  \if@minipage\else
   \advance \leftskip by \verbatimleftmargin
  \fi
  \rightskip\z@skip
  \parindent\z@\parfillskip\@flushglue\parskip\z@skip
  \@@par
  \@tempswafalse
  \def\par{%
    \if@tempswa
      \leavevmode \null \@@par\penalty\interlinepenalty
    \else
      \@tempswatrue
      \ifhmode\@@par\penalty\interlinepenalty\fi
    \fi}%
  \let\do\@makeother \dospecials
  \obeylines 
   \verbatimsize \baselineskip\verbatimbaselineskip
   \ttfamily \@noligs
  \everypar \expandafter{\the\everypar \unpenalty}%
}

%% from latex.ltx
%%  v1.3
\def\@item[#1]{% 
  \if@noparitem
    \@donoparitem
  \else
    \if@inlabel
      \indent \par
    \fi
    \ifhmode
      \unskip\unskip \par
    \fi
    \if@newlist
      \if@nobreak
        \@nbitem
      \else
        \addpenalty\@beginparpenalty
        \addvspace\@topsep
        \addvspace{-\parskip}%
      \fi
    \else
      \addpenalty\@itempenalty
      \addvspace\itemsep
    \fi
    \global\@inlabeltrue
  \fi
  \everypar{%
    \@minipagefalse
    \global\@newlistfalse
    \if@inlabel
      \global\@inlabelfalse
      {\setbox\z@\lastbox
       \ifvoid\z@
         \kern-\itemindent
       \fi}%
      \box\@labels
      \penalty\z@
    \fi
    \if@nobreak
      \@nobreakfalse
      \clubpenalty\z@ %\@M --> \z@ 
    \else
      \clubpenalty \@clubpenalty
      \everypar{}%
    \fi}%
  \if@noitemarg
    \@noitemargfalse
    \if@nmbrlist
      \refstepcounter\@listctr
    \fi
  \fi
  \sbox\@tempboxa{\makelabel{#1}}%
  \global\setbox\@labels\hbox{%
    \unhbox\@labels
    \hskip \itemindent
    \hskip -\labelwidth
    \hskip -\labelsep
    \ifdim \wd\@tempboxa >\labelwidth
      \box\@tempboxa
    \else
      \hbox to\labelwidth {\unhbox\@tempboxa}%
    \fi
    \hskip \labelsep}%
  \ignorespaces\inhibitglue
}

\newcounter{figure}
\renewcommand{\thefigure}{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\thefigure}
\def\fnum@efigure{Fig.\,\thefigure}
\newenvironment{figure}
               {\floatenv\capwidth\hsize
                \abovecaptionskip1mm\belowcaptionskip\z@\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\floatenv\capwidth35zw
                \abovecaptionskip1mm\belowcaptionskip\z@\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand{\thetable}{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\thetable}
\def\fnum@etable{Table~\thetable}
\newenvironment{table}
               {\floatenv\capwidth\hsize
                \abovecaptionskip\z@\belowcaptionskip1mm\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\floatenv\capwidth35zw
                \abovecaptionskip\z@\belowcaptionskip1mm\@dblfloat{table}}
               {\end@dblfloat}
\def\@floatboxreset {%
        \reset@font
        \footnotesize\tabcolsep.5zw
        \@setminipage
}
\def\floatenv{%
 \renewcommand{\footnotesize}{%
  \@setfontsize\footnotesize\@viiipt{16\@Q}%
  \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
  \abovedisplayshortskip \z@ \@plus\p@
  \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
  \def\@listi{\leftmargin\leftmargini
              \topsep 3\p@ \@plus\p@ \@minus\p@
              \parsep 2\p@ \@plus\p@ \@minus\p@
              \itemsep \parsep}%
  \belowdisplayskip \abovedisplayskip}%
 \footnotesize
 \let\center\Center
}

\def\Center{\topsep\z@\partopsep\z@\parsep\z@\itemsep\z@
 \trivlist \centering\item\relax}
\def\endCenter{\endtrivlist}

\long\def\@caption#1[#2]#3{\par
% \addcontentsline{\csname ext@#1\endcsname}{#1}%
%  {\protect\numberline{\csname the#1\endcsname}{\ignorespaces #2}}%
 \begingroup
  \@parboxrestore
  \if@minipage
   \@setminipage
  \fi
   \small\baselineskip14\@Q
   \@makecaption{\csname fnum@#1\endcsname}{\ignorespaces #3}\par
 \endgroup}

\newlength\abovecaptionskip
\newlength\belowcaptionskip

\newbox\@tempboxb
\newdimen\capwidth
\newdimen\ecapwidth
\def\capnum@font{\gtfamily\sffamily}

\long\def\@makecaption#1#2{%
 \vskip\abovecaptionskip
 \setbox\@tempboxa\hbox{%
   \def\@ixstackcr{\@ifnextchar[% ]
    \@istackcr{\cr\hskip1zw\ignorespaces}}%
   \def\@istackcr[#1]{\cr\noalign{\vskip #1}\hskip1zw\ignorespaces}%
  \shortstack[l]{{\capnum@font #1}\hskip1zw #2}}%
 \@tempdima \ht\@tempboxa
 \advance \@tempdima \dp\@tempboxa
 \ifdim \@tempdima > \baselineskip
  \ifdim \wd\@tempboxa > \capwidth
   \centering\parbox[t]{\capwidth}{\hangindent1zw
    {\capnum@font #1}\hskip1zw #2}\par
  \else
   \centering\parbox[t]{\wd\@tempboxa}{\hangindent1zw
    {\capnum@font #1}\hskip1zw #2}\par
  \fi
 \else
  \ifdim \wd\@tempboxa > \capwidth
   \centering\parbox[t]{\capwidth}{\hangindent1zw
    {\capnum@font #1}\hskip1zw #2}\par
  \else
   \hbox to\hsize{\hss\box\@tempboxa\hss}\par
  \fi
 \fi
 \vspace{\belowcaptionskip}%
}

\long\def\ecaption#1{\small
 \setbox\@tempboxa\hbox{\csname fnum@e\@captype\endcsname\hskip1zw
  \shortstack[l]{#1}}%
 \@tempdima\ht\@tempboxa \advance \@tempdima \dp\@tempboxa
 \setbox\@tempboxb\hbox{\csname fnum@e\@captype\endcsname\hskip1zw}%% <<
 \ifdim \@tempdima > \baselineskip
  \ifdim \wd\@tempboxa > \ecapwidth
   \hfil\parbox[t]{\ecapwidth}{\hangindent\wd\@tempboxb 
    \csname fnum@e\@captype\endcsname\hskip1zw #1}\par
  \else
   \hfil\parbox[t]{\wd\@tempboxa}{\hangindent\wd\@tempboxb 
    \csname fnum@e\@captype\endcsname\hskip1zw #1}\par
  \fi
 \else %% <
  \ifdim \wd\@tempboxa > \ecapwidth
   \hfil\parbox[t]{\ecapwidth}{\hangindent\wd\@tempboxb
    \csname fnum@e\@captype\endcsname\hskip1zw #1}\par
  \else
   \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
 \fi
 \vspace{\belowcaptionskip}\par
}

\arraycolsep 5\p@
\tabcolsep .5zw
\arrayrulewidth .12mm
\doublerulesep 2\p@
\tabbingsep \labelsep

\skip\@mpfootins = \skip\footins
\setlength\fboxsep{.5zw}
\setlength\fboxrule{.12mm}
\renewcommand{\theequation}{\@arabic\c@equation}
\if@NTT\else
 \if@enablejfam
  \DeclareSymbolFont{mincho}{JY1}{mc}{m}{n}
  \DeclareSymbolFontAlphabet{\mathmc}{mincho}
  \SetSymbolFont{mincho}{bold}{JY1}{gt}{m}{n}
  \DeclareMathAlphabet\mathgt{JY1}{gt}{m}{n}
  \reDeclareMathAlphabet{\mathrm}{\@mathrm}{\@mathmc}
  \reDeclareMathAlphabet{\mathbf}{\@mathbf}{\@mathgt}
  \jfam\symmincho
 \else
  \DeclareRobustCommand{\mathmc}{%
    \@latex@error{Command \noexpand\mathmc invalid with\space
       `disablejfam' class option.}\@eha
  }
  \DeclareRobustCommand{\mathgt}{%
    \@latex@error{Command \noexpand\mathgt invalid with\space
       `disablejfam' class option.}\@eha
  }
 \fi
\fi

%% for 2.1.4 compatibility
\@ifundefined{reDeclareMathAlphabet}{%
 \def\reDeclareMathAlphabet#1#2#3{%
  \DeclareRobustCommand{#1}[1]{%
    \ifmmode
      \ifx\math@bgroup\@empty % 2.09 compatibility
        #2\relax#3\relax##1\relax
      \else % native mode
        \ifx\math@bgroup\relax % oldstyle
          #2\relax\@fontswitch\relax{#3}##1\relax
        \else
          #2{#3{##1}}%
        \fi
      \fi
    \else
      #2{##1}%
    \fi
  }%
 }%
 \@onlypreamble\reDeclareMathAlphabet
 }{}

\@ifundefined{mathmc}{\relax}{%
 \if@enablejfam
  \DeclareSymbolFont{mincho}{JY1}{mc}{m}{n}%
  \DeclareSymbolFontAlphabet{\mathmc}{mincho}%
  \SetSymbolFont{mincho}{bold}{JY1}{gt}{m}{n}%
  \DeclareMathAlphabet\mathgt{JY1}{gt}{m}{n}%
  \reDeclareMathAlphabet{\mathrm}{\@mathrm}{\@mathmc}%
  \reDeclareMathAlphabet{\mathbf}{\@mathbf}{\@mathgt}%
  \jfam\symmincho
 \else
  \DeclareRobustCommand{\mathmc}{%
    \@latex@error{Command \noexpand\mathmc invalid with\space
       `disablejfam' class option.}\@eha
  }%
  \DeclareRobustCommand{\mathgt}{%
    \@latex@error{Command \noexpand\mathgt invalid with\space
       `disablejfam' class option.}\@eha
  }%
 \fi
}

\@ifundefined{mathmc}{\relax}{%
 \DeclareOldFontCommand{\mc}{\normalfont\mcfamily}{\mathmc}%
 \DeclareOldFontCommand{\gt}{\normalfont\gtfamily}{\mathgt}%
}

\if@NTT\else
 \DeclareOldFontCommand{\mc}{\normalfont\mcfamily}{\mathmc}
 \DeclareOldFontCommand{\gt}{\normalfont\gtfamily}{\mathgt}
\fi
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*{\cal}{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*{\mit}{\@fontswitch\relax\mathnormal}

\newdimen\bibindent
\setlength\bibindent{1.5em}
\newcommand{\newblock}{\hskip .11em\@plus.33em\@minus.07em}

\newenvironment{thebibliography}[1]
 {%
  \renewcommand{\section}{\@startsection{section}{1}{\z@}%
   {.5\Cvs \@plus\z@ \@minus.1\Cdp}% 
   {.5\Cvs \@plus\z@ \@minus.1\Cdp}%
   {\reset@font\bfseries\normalsize\centering}}%
  \section*{\refname}%
  \footnotesize\baselineskip13\@Q
  %%
  \afterassignment\remove@to@nnil
   \count@ 0#1\relax\@nnil
  \ifnum\count@=\z@
   \global\def\twoorthree@digits##1{%
    \setbox0\hbox{99}\hbox to \wd0{\hss ##1\hss}}%
  \else
   \ifnum#1>99\relax
    \global\def\twoorthree@digits##1{%
     \setbox0\hbox{999}\hbox to \wd0{\hss ##1\hss}}%
   \else
    \global\def\twoorthree@digits##1{%
     \setbox0\hbox{99}\hbox to \wd0{\hss ##1\hss}}%
   \fi
  \fi
  %%
   \list{\@biblabel{\@arabic\c@enumiv}}%
        {%
         \topsep\z@\partopsep\z@\parsep\z@\itemsep\z@\parskip\z@
         \settowidth\labelwidth{\@biblabel{#1}}% 
         \leftmargin1zw
         \labelsep1zw
         \itemindent\labelwidth
         \@openbib@code
         \usecounter{enumiv}%
         \let\p@enumiv\@empty
         \renewcommand\theenumiv{\@arabic\c@enumiv}}%
   \sloppy
   \clubpenalty\z@
   \@clubpenalty\clubpenalty
   \widowpenalty\z@
   \sfcode`\.\@m}
  {\def\@noitemerr
    {\@latex@warning{Empty `thebibliography' environment}}%
   \endlist}
\def\@biblabel#1{[\twoorthree@digits{#1}]}
\let\@openbib@code\@empty

\renewcommand{\footnoterule}{%
  \kern-3\p@
  \hrule height .1mm width \columnwidth
  \kern 2.715\p@}

\def\@makefnmark{\hbox{$^{\mbox{\tiny \kern-.5zw
  （注\kern.1zw\@thefnmark{}）\kern-.5zw}}$}}

\newcommand\@makefntext[1]{\parindent1zw
 \noindent\hbox{\kern-.5zw
  （注\kern.1zw\@thefnmark{}）\kern-.5zw\hbox to 1zw{\hss ：\hss}}#1}

\def\thefootnote{\@arabic\c@footnote}
\def\thempfootnote{\@fnsymbol\c@mpfootnote}

%% from [proc|slide].cls
%% \def\thefootnote{\fnsymbol{footnote}}
%% from [article|report|book].cls
%% \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
%% from latex.ltx
%% \def\@fnsymbol#1{\ensuremath{\ifcase#1\or *\or \dagger\or \ddagger\or
%%    \mathsection\or \mathparagraph\or \|\or **\or \dagger\dagger
%%    \or \ddagger\ddagger \else\@ctrerr\fi}}

%% from fleqn.clo
%%\ProvidesFile{fleqn.clo}
%%       [1998/08/17 v1.1c Standard LaTeX option
%%                                   (flush left equations)]
\newdimen\mathindent
\AtEndOfClass{\mathindent1zw}% \leftmargini
\renewcommand\[{\relax
                \ifmmode\@badmath
                \else
                 \parsep  4\p@ \@plus2\p@ \@minus\p@
                 \topsep  8\p@ \@plus2\p@ \@minus4\p@
                 \itemsep 4\p@ \@plus2\p@ \@minus\p@
                  \begin{trivlist}%
                    \@beginparpenalty\predisplaypenalty
                    \@endparpenalty\postdisplaypenalty
                    \item[]\leavevmode
                    \hb@xt@\linewidth\bgroup $\m@th\displaystyle %$
                      \hskip\mathindent\bgroup
                \fi}
\renewcommand\]{\relax
                \ifmmode
                      \egroup $\hfil% $
                    \egroup
                  \end{trivlist}%
                \else \@badmath
                \fi}
\renewenvironment{equation}%
    {\@beginparpenalty\predisplaypenalty
     \@endparpenalty\postdisplaypenalty
     \refstepcounter{equation}%
     \parsep  4\p@ \@plus2\p@ \@minus\p@
     \topsep  8\p@ \@plus2\p@ \@minus4\p@
     \itemsep 4\p@ \@plus2\p@ \@minus\p@
     \trivlist \item[]\leavevmode
       \hb@xt@\linewidth\bgroup $\m@th% $
         \displaystyle
         \hskip\mathindent}%
        {$\hfil % $
         \displaywidth\linewidth\hbox{\@eqnnum}%
       \egroup
     \endtrivlist}
\renewenvironment{eqnarray}{%
    \stepcounter{equation}%
    \parsep  4\p@ \@plus2\p@ \@minus\p@
    \topsep  8\p@ \@plus2\p@ \@minus4\p@
    \itemsep 4\p@ \@plus2\p@ \@minus\p@
    \def\@currentlabel{\p@equation\theequation}%
    \global\@eqnswtrue\m@th
    \global\@eqcnt\z@
    \tabskip\mathindent
    \let\\=\@eqncr
    \setlength\abovedisplayskip{\topsep}%
    \ifvmode
      \addtolength\abovedisplayskip{\partopsep}%
    \fi
    \addtolength\abovedisplayskip{\parskip}%
    \setlength\belowdisplayskip{\abovedisplayskip}%
    \setlength\belowdisplayshortskip{\abovedisplayskip}%
    \setlength\abovedisplayshortskip{\abovedisplayskip}%
    $$\everycr{}\halign to\linewidth% $$
    \bgroup
      \hskip\@centering
      $\displaystyle\tabskip\z@skip{##}$\@eqnsel&%
      \global\@eqcnt\@ne \hskip.8\arraycolsep \hfil${##}$\hfil&%
      \global\@eqcnt\tw@ \hskip.8\arraycolsep
        $\displaystyle{##}$\hfil \tabskip\@centering&%
      \global\@eqcnt\thr@@
        \hb@xt@\z@\bgroup\hss##\egroup\tabskip\z@skip\cr}%
      {\@@eqncr
    \egroup
    \global\advance\c@equation\m@ne$$% $$
    \@ignoretrue
    }
%%\endinput
%%
%% End of file `fleqn.clo'.

\def\@eqnnum{{\normalfont \normalcolor (\theequation)\hspace*{3mm}}}

\def\@begintheorem#1#2{%
 \settowidth{\labelwidth}{\kern-.5zw ［#1\ #2］\kern-.5zw}
 \begin{description}
  \item[\@thmlbracket #1\ #2\@thmrbracket]}

\def\@opargbegintheorem#1#2#3{%
 \begin{description}
  \item[\@thmlbracket #1\ #2\@thmrbracket （#3）\kern-.5zw]}%%
\def\@endtheorem{\end{description}}

\def\thmbracket#1#2{\gdef\@thmlbracket{#1}\gdef\@thmrbracket{#2}}
\thmbracket{\kern-.5zw ［}{］\kern-.5zw }
\def\@thmcountersep{}

\def\ack{\par{\bfseries 謝辞}\hskip1zw}

\if@NTT\else
\newif\if西暦 \西暦false
\def\西暦{\西暦true}
\def\和暦{\西暦false}
\def\today{{%
  \iftdir
    \if西暦
      \kansuji\number\year 年
      \kansuji\number\month 月
      \kansuji\number\day 日
    \else \@tempcnta\year
      \advance\@tempcnta-1988 平成
      \ifnum \@tempcnta =1 元\else\kansuji\number\@tempcnta\fi 年
      \kansuji\number\month 月\kansuji\number\day 日
    \fi
  \else
    \if西暦
      \number\year\, 年\hspace{.5em}\nobreak
      \number\month\, 月\hspace{.5em}\nobreak
      \number\day\, 日
    \else \@tempcnta\year
      \advance\@tempcnta-1988 平成
      \ifnum \@tempcnta =1 元\else\number\@tempcnta\fi
        \, 年\hspace{.5em}\nobreak
      \number\month\, 月\hspace{.5em}\nobreak\number\day\, 日
    \fi
  \fi}}
\fi
%%\newcommand{\prepartname}{第}
%%\newcommand{\postpartname}{部}
%%\newcommand{\contentsname}{目 次}
%%\newcommand{\listfigurename}{図 目 次}
%%\newcommand{\listtablename}{表 目 次}
\newcommand{\refname}{参考文献}
%%\newcommand{\indexname}{索 引}
\newcommand{\figurename}{図--}
\newcommand{\tablename}{表--}
%%\newcommand{\appendixname}{付 録}
%%\newcommand{\abstractname}{概 要}


\def\figref{\@ifstar
 {\nf@figref}{\bf@figref}}
\def\bf@figref#1{%
 \@ifundefined{BF@#1}%
  {\expandafter\def\csname BF@#1\endcsname{}%
   \leavevmode
   {\gtfamily\sffamily \figurename\nobreak\ref{#1}}}%
  {\figurename\nobreak\ref{#1}}%
}
\def\nf@figref#1{\figurename\nobreak\ref{#1}}

\def\tabref{\@ifstar
 {\nf@tabref}{\bf@tabref}}
\def\bf@tabref#1{%
 \@ifundefined{BF@#1}%
  {\expandafter\def\csname BF@#1\endcsname{}%
   \leavevmode
   {\gtfamily\sffamily \tablename\nobreak\ref{#1}}}%
  {\tablename\nobreak\ref{#1}}%
}
\def\nf@tabref#1{\tablename\nobreak\ref{#1}}




%%## citesort.sty ##%%
\newcount\@minsofar
\newcount\@min
\newcount\@cite@temp
\def\@citex[#1]#2{%
\if@filesw \immediate \write \@auxout {\string \citation {#2}}\fi
\@tempcntb\m@ne \let\@h@ld\relax \def\@citea{}%
\@min\m@ne%
\@cite{%
  \@for \@citeb:=#2\do {\@ifundefined {b@\@citeb}%
    {\@h@ld\@citea\@tempcntb\m@ne{\bfseries ?}%
    \@warning {Citation `\@citeb ' on page \thepage \space undefined}}%
{\@minsofar\z@ \@for \@scan@cites:=#2\do {%
  \@ifundefined{b@\@scan@cites}%
    {\@cite@temp\m@ne}
    {\@cite@temp\number\csname b@\@scan@cites \endcsname \relax}%
\ifnum\@cite@temp > \@min% select the next one to list
    \ifnum\@minsofar = \z@
      \@minsofar\number\@cite@temp
      \edef\@scan@copy{\@scan@cites}\else
    \ifnum\@cite@temp < \@minsofar
      \@minsofar\number\@cite@temp
      \edef\@scan@copy{\@scan@cites}\fi\fi\fi}\@tempcnta\@min
  \ifnum\@minsofar > \z@ % some more
    \advance\@tempcnta\@ne
    \@min\@minsofar
    \ifnum\@tempcnta=\@minsofar %   Number follows previous--hold on to it
      \ifx\@h@ld\relax
        \edef \@h@ld{\@citea\csname b@\@scan@copy\endcsname}%
      \else \edef\@h@ld{\ifmmode{-}\else--\fi\csname b@\@scan@copy\endcsname}%
      \fi
    \else \@h@ld\@citea\csname b@\@scan@copy\endcsname
          \let\@h@ld\relax
  \fi % no more
\fi}%
\def\@citea{,\penalty\@highpenalty\,}}\@h@ld}{#1}}

%% from "cite.sty", with a little modified
\def\@cite#1#2{\leavevmode %% \unskip
  \ifnum\lastpenalty=\z@\penalty\@highpenalty\fi % highpenalty before
  [{\multiply\@highpenalty 3 #1%             % triple-highpenalties within
      \if@tempswa,\penalty\@highpenalty\ #2\fi % and before note.
    }]\spacefactor\@m}
%% end of cite.sty

\let\org@cite\@cite
\def\cite{\@ifnextchar[% ]
 {\@tempswatrue\let\@cite\org@cite\@citex}
 {\@tempswafalse\let\@cite\org@cite\@citex[]}}

\def\onelineskip{\par\vspace{1\baselineskip}\par}
\def\halflineskip{\par\vspace{.5\baselineskip}\par}

\def\ddash{\hbox to 2zw{---\hss ---}}% \let\doubledash\ddash
\def\FRAC#1#2{\leavevmode\kern-.1em
  \raise.5ex\hbox{\the\scriptfont0 #1}\kern-.1em
  /\kern-.15em\lower.25ex\hbox{\the\scriptfont0 #2}}
\def\RN#1{\uppercase\expandafter{\romannumeral#1}}

\def\MARU#1{\raisebox{0.1zh}{{%
 \leavevmode\ooalign{\hfil\lower.168ex\hbox{#1}\hfil\crcr\mathhexbox20D}}}}

\def\kintou#1#2{\@ifundefined{jintercharskip}%
 {\leavevmode\hbox to #1{%
   \kanjiskip\z@ \@plus 1fill \@minus 1fill\xkanjiskip\kanjiskip #2}}%
 {\leavevmode\hbox to #1{\jintercharskip\z@ \@plus 1fill \@minus 1fill
   \jasciikanjiskip\jintercharskip #2}}}

\def\ruby#1#2{\leavevmode \setbox0=\hbox{#1}\setbox1=\hbox{\tiny #2}%% <
 \ifdim\wd0>\wd1 \dimen0=\wd0 \else \dimen0=\wd1 \fi
 \hbox{\kanjiskip=\fill
 \vbox{\hbox to \dimen0{\tiny \hfil#2\hfil}%
  \nointerlineskip \hbox to \dimen0{\hfil#1\hfil}}}}

\def\QED{{\unskip\nobreak\hfil\penalty50
 \hskip1zw\hbox{}\nobreak\hfil \hbox{$\Box$\hskip1zw}
 \parfillskip\z@ \finalhyphendemerits\z@\par}}

\def\Hline{\noalign{\hrule height 0.4mm}}

\def\leq{\mathrel{\mathpalette\gl@align<}} \let\le\leq
\def\geq{\mathrel{\mathpalette\gl@align>}} \let\ge\geq
\def\gl@align#1#2{\lower.6ex\vbox{\baselineskip\z@skip\lineskip\z@
 \ialign{$\m@th#1\hfil##\hfil$\crcr#2\crcr=\crcr}}}
\def\subseteq{\rlap{\raisebox{.25ex}{$\subset$}}\raisebox{-1.1ex}{$=$}}
\def\supseteq{\rlap{\raisebox{.25ex}{$\supset$}}\raisebox{-1.1ex}{$=$}}
\def\neq{\mathrel{\mathpalette\c@ncelass=}}
\def\c@ncelass#1#2{\ooalign{$\hfil#1\mkern-.4mu|\hfil$\crcr$#1#2$}}
\let\ne\neq

%\if@NTT\else
% \postbreakpenalty`（=10000
% \prebreakpenalty`）=10000
% \postbreakpenalty`［=10000
% \prebreakpenalty`］=10000
% \postbreakpenalty`〈=10000
% \prebreakpenalty`〉=10000
% \prebreakpenalty`"=10000
% \prebreakpenalty`℃=10000
% \prebreakpenalty`　=10000
% \prebreakpenalty`・=10000
% \prebreakpenalty`\UTF{FF5E}=10000
% \prebreakpenalty`ー=10000
% \prebreakpenalty`？=10000
% \prebreakpenalty`！=10000
% \postbreakpenalty"5C=10000
%\fi

\def\sloppy{\tolerance9999\hbadness9999\hfuzz.5\p@ \vfuzz.5\p@}

\hbadness4000
\vbadness10001
\tolerance4000\pretolerance\m@ne
\clubpenalty\z@
\widowpenalty\z@
\displaywidowpenalty\z@
\interdisplaylinepenalty\z@ %%  100
\predisplaypenalty\z@       %% 1000

\AtBeginDocument{%
 \@ifundefined{@mathmargin}{\let\@mathmargin\undefined}
  {%
%   \let\@mathmargin\mathindent
   \@mathmargin1zw \@plus\z@ \@minus\z@
   \def\maketag@@@#1{\hbox{\m@th\normalfont #1}\hskip3mm}%
   \def\eqref#1{\textup{\tagform@{\ref{#1}}\unskip}}%
%   %% from /usr/local/share/texmf/tex/latex/amslatex/amsmath.sty (v1.2c)
%   \renewenvironment{equation*}{%
%    \equation \nonumber
%   }{%
%    % \nonumber
%    \endequation
%   }%
  }%
 %% here.sty
 \@ifundefined{@HHfloat}{}{%
  \def\@HHfloat#1[H]{%
   \expandafter\let\csname end#1\endcsname\end@Hfloat
   \vskip\intextsep
   \vbox\bgroup\def\@captype{#1}\parindent\z@
   \@floatboxreset %%
   \ignorespaces}%
  \def\end@Hfloat{\egroup\vskip-1zh %%
   \vskip \intextsep}%
 }%
 %% nidanfloat
 \@ifundefined{dblbotfraction}{}{%
  \renewcommand\dblbotfraction{0.9}% 0.5
  \setcounter{dblbotnumber}{4}% 2
 }%
 %%
}

\pagestyle{empty}

\pagenumbering{arabic}
\raggedbottom
\if@twocolumn
  \twocolumn
  \sloppy
   \raggedbottom
\else
  \onecolumn
\fi
\if@twoside
  \@mparswitchtrue
\else
  \@mparswitchfalse
\fi

\endinput
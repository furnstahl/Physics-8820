# Overview of Learning from Data

These lectures notes are for a course on Bayesian statistics for physicists.
This means that we will not only use physics examples but we will build on physics intuition and the particular mathematical skills that every physicist acquires.

The course introduces a variety of central algorithms and methods essential for performing scientific data analysis using statistical inference and machine learning. The emphasis is on practical applications of Bayesian inference in physics, i.e. the ability to quantify the strength of inductive inference from facts (such as experimental data) to propositions such as scientific hypotheses and models.

The course is project-based, and students will be exposed to some fundamental research problems through the various projects, with the aim to reproduce state-of-the-art scientific results. Students will use Jupyter notebooks for scientific data analysis examples, with Python as a programming language and including relevant open-source libraries.

## Instructors
* _Lecturer_: Dick Furnstahl
  * _Email_: furnstahl.1@osu.edu
  * _Office_: OSU Department of Physics, PRB, room M2048
* _Teaching assistant_: *TBA*
  * _Email_: *TBA* 
  * _Office_: *TBA* 
  
<!-- !split -->

## Objectives

In recent years there has been an explosion of interest in the use of Bayesian methods in many sub-fields of physics. In nuclear physics (my specialty), these methods are being used to quantify the uncertainties in theoretical work on topics ranging from the inter-nucleon forces to high-energy heavy-ion collisions, develop more reliable extrapolants of nuclear-energy-density functionals towards the limits of nuclear existence, predict the impact that future NICER observations may have on the equation of state of neutron matter, and determine whether or not nucleon resonances are present in experimental data.
Meanwhile machine learning is gaining increased currency as a method for identifying interesting signals in both experiments and simulations. 

While most physics Ph.D. students are taught some standard (frequentist) statistics as part of their course work, relatively few encounter Bayesian methods until they are engaged in research. But Bayesian methods provide a coherent and compelling framework to think about inference, and so can be applied to many important questions in physics. The overall learning goal of this course is to take students who have had little (or no) previous exposure to Bayes’ theorem and show them how it can be applied to problems of parameter estimation, model selection, and machine learning. Jupyter notebooks will be used extensively throughout the course, so this is also a great opportunity to learn some Python and the use of these notebooks. There will be several guided "mini-projects" on topics including Bayesian parameter estimation, model selection, Bayesian optimization, and Bayesian neutral networks. You will be asked to put together a "project" based on your own physics interests and using methods from the course. Grading will be based on Juypter-based problems and these projects.

## Prerequisites

The course will only assume statistics knowledge at the level taught in undergraduate physics labs (e.g., least-squares fitting), and physics and mathematics knowledge at the advanced undergraduate level (e.g., basic quantum mechanics, linear algebra, and vector calculus).


## Learning outcomes
Upon completion of this course students should be able to:

- Apply the rules of probability to derive posterior probability distributions for simple problems involving prior information on parameters and various standard likelihood functions.
- Perform Bayesian parameter estimation, including in cases where marginalization over nuisance parameters is required.
- Use Monte Carlo sampling to generate posterior probability distributions and identify problems where standard sampling is likely to fail.
- Compute an evidence ratio and explain what it means.
- Explain machine learning from a Bayesian perspective and employ a testing and training data set to develop and validate a Gaussian-process model.
- Employ these methods in the context of specific physics problems (examples in class will often be taken from nuclear physics, but they have more general applicability).
- Be able to understand, appreciate, and criticize the growing literature on Bayesian statistics and machine learning for physics applications.

## Topics

The following topics will be covered (this is not an exclusive list):

- Basics of Bayesian statistics
- Bayesian parameter estimation
- Why Bayes is better
- MCMC sampling
- Assigning probabilities
- Model selection
- Model checking
- Gaussian processes
- Special topic: Bayesian methods and machine learning. [Note: we will not cover machine learning in great detail, but learn about connections to Bayesian methods, e.g., with Bayesian neural networks as a working example.]
- Special topic: emulators



<!-- ======= Acknowledgements ======= -->

## Teaching philosophy

In simplest terms, the pedagogical approach is analogous to learning to swim by being thrown in the water. Some features:

* you absorb details as we go through interactive notebooks in class rather than by listening exclusively to lectures;
* a spiral method is used: we do not introduce all elements of a topic at once, but keep coming back to a topic at increasing levels of sophistication (or resolution, to use another physics analogy);
* the spirit is much more like fundamental research than traditional physics pedagogy.

This approach can be confusing and sometimes frustrating. But it works (with validation from physics education research) and is good training for real-world situations and for learning how to learn, so you can comfortably extend your statistics knowledge on your own. To get the most out of the course, be sure to:

* ask questions;
* question authority (including the lecturer and what you read in the literature);
* experiment;
* verify everything.

## Moral qualities of the scientist

Adapted from G. Polya, Induction and Analogy in Mathematics, chapter 1, section 4, which is entitled "The Inductive Attitude".

Intellectual courage
: One should be ready to revise any one of our beliefs.

Intellectual honesty
: One should change a belief when there is a compelling reason to do so. To stick to a conjecture clearly contradicted by experience is dishonest. It is also dishonest to ignore information or not to state and criticize all assumptions.

Wise restraint
: One should not change a belief wantonly, without some good reason. Don't just follow fashion. Do not believe anything, but question only what is worth questioning.




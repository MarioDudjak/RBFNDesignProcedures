using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Nelder_Mead
{
    // A.R. Conn, K. Scheinberg, L.N. Vicente,
    // Introduction to derivative-free optimization,
    // Siam, 2009
    class NMalgorithm
    {
        private double[][] mSolutions;
        private double[] mCostSolutions;

        private double[][] mShrikedSolutions;
        private double[] mCostShrinkedSolutions;

        private double[] mBestFoundSolution;
        private double mCostBestFoundSolution;

        private double mGammaS;
        private double mDeltaIC, mDeltaOC, mDeltaR, mDeltaE;

        private int mIndexBest;
        private int mIndexWorst, mIndexSecondWorst; 

        private int mDimensionality;
        private int mCountSolutions;

        private int mMaxNFEs;
        private int mNFEs;

        private double mMinBound, mMaxBound;

        private Random mRndGen;

        public NMalgorithm(int Dimensionality, int MaxNFEs, 
            double GammaS, double DeltaIC, double DeltaOC, double DeltaR, double DeltaE, 
            double MinBound, double MaxBound, Random RndGen)
        {
            this.mDimensionality = Dimensionality;
            this.mCountSolutions = this.mDimensionality + 1;

            this.mGammaS = GammaS;
            this.mDeltaIC = DeltaIC;
            this.mDeltaOC = DeltaOC;
            this.mDeltaR = DeltaR;
            this.mDeltaE = DeltaE;

            this.mIndexBest = 0;
            this.mIndexSecondWorst = this.mDimensionality - 1;
            this.mIndexWorst = this.mDimensionality;

            this.mSolutions = this.CreateMatrix(this.mCountSolutions, this.mDimensionality);
            this.mCostSolutions = new double[this.mCountSolutions];
            this.mShrikedSolutions = this.CreateMatrix(this.mCountSolutions, this.mDimensionality);
            this.mCostShrinkedSolutions = new double[this.mCountSolutions];
            this.mBestFoundSolution = new double[this.mDimensionality];

            this.mMaxNFEs = MaxNFEs;
            this.mNFEs = 0;

            this.mMinBound = MinBound;
            this.mMaxBound = MaxBound;

            this.mRndGen = RndGen;
        }

        public double Run()
        {
            this.InitializeSolutions();
            this.SortSolutions();

            double[] solutionYR = new double[this.mDimensionality];
            double costSolutionYR;
            double[] solutionYE = new double[this.mDimensionality];
            double costSolutionYE;
            double[] solutionYOC = new double[this.mDimensionality];
            double costSolutionYOC;
            double[] solutionYIC = new double[this.mDimensionality];
            double costSolutionYIC;

            Array.Copy(this.mSolutions[this.mIndexBest], this.mBestFoundSolution, this.mDimensionality);
            this.mCostBestFoundSolution = this.mCostSolutions[this.mIndexBest];

            double[] centroid;
            while (this.mNFEs < this.mMaxNFEs)
            {
                //Ordering
                this.SortSolutions();
                if (this.mCostSolutions[this.mIndexBest] < this.mCostBestFoundSolution)
                {
                    Array.Copy(this.mSolutions[this.mIndexBest], this.mBestFoundSolution, this.mDimensionality);
                    this.mCostBestFoundSolution = this.mCostSolutions[this.mIndexBest];
                }
                //Reflection
                centroid = this.CalcCentroid();
                for (int i = 0; i < this.mDimensionality; i++)
                    solutionYR[i] = centroid[i] + this.mDeltaR * (centroid[i] - this.mSolutions[this.mIndexWorst][i]);
                costSolutionYR = this.EvaluateSolution(solutionYR);
                this.mNFEs++;
                if (costSolutionYR >= this.mCostSolutions[this.mIndexBest] && costSolutionYR < this.mCostSolutions[this.mIndexSecondWorst])
                {
                    Array.Copy(solutionYR, this.mSolutions[this.mIndexWorst], this.mDimensionality);
                    this.mCostSolutions[this.mIndexWorst] = costSolutionYR;
                    continue;
                }
                if (this.mNFEs >= this.mMaxNFEs)
                    break;

                //Expansion
                if (costSolutionYR < this.mCostSolutions[this.mIndexBest])
                {
                    for (int i = 0; i < this.mDimensionality; i++)
                        solutionYE[i] = centroid[i] + this.mDeltaE * (centroid[i] - this.mSolutions[this.mIndexWorst][i]);
                    costSolutionYE = this.EvaluateSolution(solutionYE);
                    this.mNFEs++;
                    if (costSolutionYE <= costSolutionYR)
                    {
                        Array.Copy(solutionYE, this.mSolutions[this.mIndexWorst], this.mDimensionality);
                        this.mCostSolutions[this.mIndexWorst] = costSolutionYE;
                        continue;
                    }
                    else
                    {
                        Array.Copy(solutionYR, this.mSolutions[this.mIndexWorst], this.mDimensionality);
                        this.mCostSolutions[this.mIndexWorst] = costSolutionYR;
                        continue;
                    }
                }
                if (this.mNFEs >= this.mMaxNFEs)
                    break;

                //Contraction
                if (costSolutionYR >= this.mCostSolutions[this.mIndexSecondWorst])
                {
                    //Outside contraction
                    if (costSolutionYR < this.mCostSolutions[this.mIndexWorst])
                    {
                        for (int i = 0; i < this.mDimensionality; i++)
                            solutionYOC[i] = centroid[i] + this.mDeltaOC * (centroid[i] - this.mSolutions[this.mIndexWorst][i]);
                        costSolutionYOC = this.EvaluateSolution(solutionYOC);
                        this.mNFEs++;
                        if (costSolutionYOC <= costSolutionYR)
                        {
                            Array.Copy(solutionYOC, this.mSolutions[this.mIndexWorst], this.mDimensionality);
                            this.mCostSolutions[this.mIndexWorst] = costSolutionYOC;
                            continue;
                        }
                        else
                        {
                            this.PerformShrinking();
                            continue;
                        }
                    }
                    if (this.mNFEs >= this.mMaxNFEs)
                        break;

                    //Inside contraction
                    if (costSolutionYR >= this.mCostSolutions[this.mIndexWorst])
                    {
                        for (int i = 0; i < this.mDimensionality; i++)
                            solutionYIC[i] = centroid[i] + this.mDeltaIC * (centroid[i] - this.mSolutions[this.mIndexWorst][i]);
                        costSolutionYIC = this.EvaluateSolution(solutionYIC);
                        this.mNFEs++;
                        if (costSolutionYIC < this.mCostSolutions[this.mIndexWorst])
                        {
                            Array.Copy(solutionYIC, this.mSolutions[this.mIndexWorst], this.mDimensionality);
                            this.mCostSolutions[this.mIndexWorst] = costSolutionYIC;
                            continue;
                        }
                        else
                        {
                            this.PerformShrinking();
                            continue;
                        }
                    }
                    if (this.mNFEs >= this.mMaxNFEs)
                        break;
                }
            }

            return this.mCostBestFoundSolution;
        }

        private void PerformShrinking()
        {
            for (int i = 0; i < this.mCountSolutions; i++)
            {
                for (int j = 0; j < this.mDimensionality; j++)
                    this.mShrikedSolutions[i][j] = this.mSolutions[this.mIndexBest][j] + this.mGammaS * (this.mSolutions[i][j] - this.mSolutions[this.mIndexBest][j]);
                this.mCostShrinkedSolutions[i] = this.EvaluateSolution(this.mShrikedSolutions[i]);
                this.mNFEs++;
                Array.Copy(this.mShrikedSolutions[i], this.mSolutions[i], this.mDimensionality);
                this.mCostSolutions[i] = this.mCostShrinkedSolutions[i];

                if (this.mCostSolutions[i] < this.mCostBestFoundSolution)
                {
                    Array.Copy(this.mSolutions[i], this.mBestFoundSolution, this.mDimensionality);
                    this.mCostBestFoundSolution = this.mCostSolutions[i];
                }
                if (this.mNFEs >= this.mMaxNFEs)
                    return;
            }
        }

        private double[] CalcCentroid()
        {
            double[] centroid = new double[this.mDimensionality];

            for (int i = 0; i < this.mDimensionality; i++)
            {
                centroid[i] = 0;
                for (int j = 0; j < this.mDimensionality; j++)
                    centroid[i] += this.mSolutions[j][i];

                centroid[i] /= this.mDimensionality;
            }

            return centroid;
        }

        private void InitializeSolutions()
        {
            for (int i = 0; i < this.mCountSolutions; i++)
            {
                for (int j = 0; j < this.mDimensionality; j++)
                    this.mSolutions[i][j] = this.mMinBound + this.mRndGen.NextDouble() * (this.mMaxBound - this.mMinBound);

                this.mCostSolutions[i] = this.EvaluateSolution(this.mSolutions[i]);
                this.mNFEs++;
            }
        }

        private double EvaluateSolution(double[] Solution)
        {
            double cost = 0;

            //Dummy function
            double tmp;
            for (int i = 0; i < this.mDimensionality; i++)
            {
                tmp = 0;
                for (int j = 0; j < i; j++)
                    tmp += Solution[j];
                cost = tmp * tmp;
            }
            return cost;
        }

        private void SortSolutions()
        {
            double[] tmpSolution = new double[this.mDimensionality];
            double tmpCost;
            for (int i = 0; i < this.mCountSolutions - 1; i++)
                for (int j = i + 1; j < this.mCountSolutions; j++)
                    if (this.mCostSolutions[i] > this.mCostSolutions[j])
                    {
                        Array.Copy(this.mSolutions[i], tmpSolution, this.mDimensionality);
                        Array.Copy(this.mSolutions[j], this.mSolutions[i], this.mDimensionality);
                        Array.Copy(tmpSolution, this.mSolutions[j], this.mDimensionality);

                        tmpCost = this.mCostSolutions[i];
                        this.mCostSolutions[i] = this.mCostSolutions[j];
                        this.mCostSolutions[j] = tmpCost;
                    }
        }

        private double[][] CreateMatrix(int R, int C)
        {
            double[][] matrix = new double[R][];
            for (int i = 0; i < R; i++)
                matrix[i] = new double[C];
            return matrix;
        }
    }
}

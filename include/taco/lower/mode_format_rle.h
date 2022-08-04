#ifndef TACO_MODE_FORMAT_RLE_H
#define TACO_MODE_FORMAT_RLE_H

#include "taco/lower/mode_format_impl.h"

namespace taco {
    class RLEModeFormat : public ModeFormatImpl {
    public:
        using ModeFormatImpl::getInsertCoord;

        RLEModeFormat();
        RLEModeFormat(bool isFull, bool isOrdered,
                                bool isUnique, bool isZeroless,
                                long long allocSize = DEFAULT_ALLOC_SIZE);

        ~RLEModeFormat() override = default;

        ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;

        ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const override;
        ModeFunction posIterAccess(std::vector<ir::Expr> pos, std::vector<ir::Expr> coords, ir::Expr values, Datatype type,
                                   Mode mode) const override;

        ModeFunction coordBounds(ir::Expr parentPos, Mode mode) const override;

        std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode,
                                        int level) const override;

        ir::Expr getWidth(Mode mode) const override;

        ir::Stmt getAppendCoord(ir::Expr pos, ir::Expr coord,
                                ir::Expr values, ir::Expr valuesCap,
                                Datatype type, Mode mode) const override;
        ir::Stmt getAppendEdges(ir::Expr parentPos, ir::Expr posBegin,
                                ir::Expr posEnd, Mode mode) const override;
        ir::Expr getSize(ir::Expr parentSize, Mode mode) const override;
        ir::Stmt getAppendInitEdges(ir::Expr parentPosBegin,
                                    ir::Expr parentPosEnd, Mode mode) const override;
        ir::Stmt getAppendInitLevel(ir::Expr parentSize, ir::Expr size,
                                    Mode mode) const override;
        ir::Stmt getAppendFinalizeLevel(ir::Expr parentSize, ir::Expr size, ir::Expr values,
                                        Mode mode) const override;


//        ModeFunction getFillRegion(ir::Expr pos, std::vector<ir::Expr> coords,
//                                   ir::Expr values, Datatype type,
//                                   Mode mode) const override;

        ir::Stmt getFillRegionAppend(ir::Expr p, ir::Expr i,
                            ir::Expr start, ir::Expr length,
                            ir::Expr run, ir::Expr values, ir::Expr valuesCap,
                            Datatype type, Mode mode) const override;

    protected:
        ir::Expr getPosArray(ModePack pack) const;

        ir::Expr getVar(std::string postfix, Mode mode, Datatype t=Int()) const;

        ir::Expr getCoordVar(Mode mode) const;
        ir::Expr getPosCoordVar(Mode mode) const;
        ir::Expr getFoundVar(Mode mode) const;
        ir::Expr getFoundCountVar(Mode mode) const;

        //Variables needed to append
        ir::Expr getFillingBoolVar(Mode mode) const;
        ir::Expr getCurrCountVar(Mode mode) const;
        ir::Expr getCurrCountValVar(Mode mode) const;


        ir::Expr getPosCapacity(Mode mode) const;

        bool equals(const ModeFormatImpl& other) const override;

        const long long allocSize;

    };
}


#endif //TACO_MODE_FORMAT_RLE_H

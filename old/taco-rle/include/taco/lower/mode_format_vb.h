#ifndef TACO_MODE_FORMAT_VB_H
#define TACO_MODE_FORMAT_VB_H

#include "taco/lower/mode_format_impl.h"

namespace taco {
    class VariableBlockModeFormat : public ModeFormatImpl {
    public:
        using ModeFormatImpl::getInsertCoord;

        VariableBlockModeFormat();
        VariableBlockModeFormat(bool isFull, bool isOrdered,
                             bool isUnique, bool isZeroless,
                             bool isLastValueFill = false,
                             long long allocSize = DEFAULT_ALLOC_SIZE);

        ~VariableBlockModeFormat() override {}

        ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;

        ModeFunction coordIterBounds(std::vector<ir::Expr> parentCoords, Mode mode) const override;
        ModeFunction coordIterAccess(ir::Expr parentPos, std::vector<ir::Expr> coords, Mode mode) const override;

        ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const override;
        ModeFunction posIterAccess(std::vector<ir::Expr> pos, std::vector<ir::Expr> coords, ir::Expr values, Datatype type,
                                   Mode mode) const override;

        ModeFunction coordBounds(ir::Expr parentPos, Mode mode) const override;

        std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode,
                                        int level) const override;

        ir::Expr getWidth(Mode mode) const override;

    protected:
        ir::Expr getPosArray(ModePack pack) const;
        ir::Expr getCoordArray(ModePack pack) const;

        ir::Expr getPosCapacity(Mode mode) const;
        ir::Expr getCoordCapacity(Mode mode) const;

        bool equals(const ModeFormatImpl& other) const override;

        const long long allocSize;
    };
}

#endif //TACO_MODE_FORMAT_VB_H
